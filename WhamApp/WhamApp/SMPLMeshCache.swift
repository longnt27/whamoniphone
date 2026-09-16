import Foundation

enum SMPLMeshCache {
    static let fileExtension = "whammesh"
    static let expectedVertexCount = 6_890

    private static let magic = Array("WHAMSMPL".utf8)
    private static let version: UInt32 = 1
    private static let componentCount: UInt32 = 3
    private static let headerSize = 24

    enum CacheError: LocalizedError {
        case invalidVertexCount(Int)
        case unexpectedVertexCount(expected: Int, actual: Int)
        case invalidFrameVertexCount(expected: Int, actual: Int)
        case nonFiniteVertex
        case alreadyFinalized
        case invalidMagic
        case unsupportedVersion(UInt32)
        case invalidComponentCount(UInt32)
        case invalidFileLength
        case frameOutOfRange(Int)

        var errorDescription: String? {
            switch self {
            case .invalidVertexCount(let count):
                return "Invalid SMPL vertex count: \(count)"
            case .unexpectedVertexCount(let expected, let actual):
                return "Expected \(expected) SMPL vertices, cache contains \(actual)"
            case .invalidFrameVertexCount(let expected, let actual):
                return "Expected \(expected) SMPL vertices, received \(actual)"
            case .nonFiniteVertex:
                return "SMPL mesh contains a non-finite coordinate"
            case .alreadyFinalized:
                return "SMPL mesh cache is already finalized"
            case .invalidMagic:
                return "Not a WHAM SMPL mesh cache"
            case .unsupportedVersion(let version):
                return "Unsupported SMPL mesh cache version: \(version)"
            case .invalidComponentCount(let count):
                return "Invalid SMPL component count: \(count)"
            case .invalidFileLength:
                return "SMPL mesh cache length does not match its header"
            case .frameOutOfRange(let index):
                return "SMPL mesh frame is out of range: \(index)"
            }
        }
    }

    static func outputURL(forJSONURL url: URL) -> URL {
        url.deletingPathExtension().appendingPathExtension(fileExtension)
    }

    final class Writer {
        let outputURL: URL
        let vertexCount: Int
        private let temporaryURL: URL
        private var handle: FileHandle?
        private(set) var frameCount = 0
        private var finalized = false

        init(outputURL: URL, vertexCount: Int = expectedVertexCount) throws {
            guard vertexCount > 0 else {
                throw CacheError.invalidVertexCount(vertexCount)
            }
            self.outputURL = outputURL
            self.vertexCount = vertexCount
            self.temporaryURL = outputURL
                .deletingLastPathComponent()
                .appendingPathComponent(
                    ".\(outputURL.lastPathComponent).\(UUID().uuidString).tmp"
                )

            guard FileManager.default.createFile(
                atPath: temporaryURL.path,
                contents: nil
            ) else {
                throw CocoaError(.fileWriteUnknown)
            }
            let handle = try FileHandle(forWritingTo: temporaryURL)
            self.handle = handle
            try handle.write(contentsOf: Self.header(
                vertexCount: vertexCount,
                frameCount: 0
            ))
        }

        deinit {
            try? handle?.close()
            if !finalized {
                try? FileManager.default.removeItem(at: temporaryURL)
            }
        }

        func append(_ vertices: [SIMD3<Float>]) throws {
            guard !finalized, let handle else {
                throw CacheError.alreadyFinalized
            }
            guard vertices.count == vertexCount else {
                throw CacheError.invalidFrameVertexCount(
                    expected: vertexCount,
                    actual: vertices.count
                )
            }

            var payload = Data(capacity: vertexCount * 3 * 2)
            for vertex in vertices {
                for coordinate in [vertex.x, vertex.y, vertex.z] {
                    guard coordinate.isFinite else {
                        throw CacheError.nonFiniteVertex
                    }
                    payload.appendLittleEndian(Float16(coordinate).bitPattern)
                }
            }
            try handle.write(contentsOf: payload)
            frameCount += 1
        }

        func finalize() throws {
            guard !finalized, let handle else {
                throw CacheError.alreadyFinalized
            }
            try handle.seek(toOffset: 16)
            var count = Data()
            count.appendLittleEndian(UInt32(frameCount))
            try handle.write(contentsOf: count)
            try handle.synchronize()
            try handle.close()
            self.handle = nil

            let fileManager = FileManager.default
            if fileManager.fileExists(atPath: outputURL.path) {
                _ = try fileManager.replaceItemAt(
                    outputURL,
                    withItemAt: temporaryURL,
                    backupItemName: nil,
                    options: []
                )
            } else {
                try fileManager.moveItem(at: temporaryURL, to: outputURL)
            }
            finalized = true
        }

        private static func header(vertexCount: Int, frameCount: Int) -> Data {
            var data = Data(magic)
            data.appendLittleEndian(version)
            data.appendLittleEndian(UInt32(vertexCount))
            data.appendLittleEndian(UInt32(frameCount))
            data.appendLittleEndian(componentCount)
            return data
        }
    }

    struct Reader {
        let vertexCount: Int
        let frameCount: Int
        private let data: Data

        init(
            url: URL,
            expectedVertexCount: Int = SMPLMeshCache.expectedVertexCount
        ) throws {
            try self.init(
                data: Data(contentsOf: url, options: .mappedIfSafe),
                expectedVertexCount: expectedVertexCount
            )
        }

        init(
            data: Data,
            expectedVertexCount: Int = SMPLMeshCache.expectedVertexCount
        ) throws {
            guard data.count >= headerSize else {
                throw CacheError.invalidFileLength
            }
            guard Array(data.prefix(magic.count)) == magic else {
                throw CacheError.invalidMagic
            }
            let storedVersion = data.littleEndianUInt32(at: 8)
            guard storedVersion == version else {
                throw CacheError.unsupportedVersion(storedVersion)
            }
            let vertexCount = Int(data.littleEndianUInt32(at: 12))
            guard vertexCount > 0 else {
                throw CacheError.invalidVertexCount(vertexCount)
            }
            guard vertexCount == expectedVertexCount else {
                throw CacheError.unexpectedVertexCount(
                    expected: expectedVertexCount,
                    actual: vertexCount
                )
            }
            let frameCount = Int(data.littleEndianUInt32(at: 16))
            let components = data.littleEndianUInt32(at: 20)
            guard components == componentCount else {
                throw CacheError.invalidComponentCount(components)
            }

            let (valuesPerFrame, valuesOverflow) = vertexCount.multipliedReportingOverflow(by: 3)
            let (bytesPerFrame, bytesOverflow) = valuesPerFrame.multipliedReportingOverflow(by: 2)
            let (payloadLength, payloadOverflow) = bytesPerFrame.multipliedReportingOverflow(by: frameCount)
            let (expectedLength, lengthOverflow) = headerSize.addingReportingOverflow(payloadLength)
            guard !valuesOverflow,
                  !bytesOverflow,
                  !payloadOverflow,
                  !lengthOverflow,
                  expectedLength == data.count else {
                throw CacheError.invalidFileLength
            }

            self.vertexCount = vertexCount
            self.frameCount = frameCount
            self.data = data
        }

        func frame(at index: Int) throws -> [SIMD3<Float>] {
            guard index >= 0 && index < frameCount else {
                throw CacheError.frameOutOfRange(index)
            }
            let frameOffset = headerSize + index * vertexCount * 3 * 2
            var vertices: [SIMD3<Float>] = []
            vertices.reserveCapacity(vertexCount)
            for vertexIndex in 0..<vertexCount {
                let offset = frameOffset + vertexIndex * 6
                let x = Float(Float16(bitPattern: data.littleEndianUInt16(at: offset)))
                let y = Float(Float16(bitPattern: data.littleEndianUInt16(at: offset + 2)))
                let z = Float(Float16(bitPattern: data.littleEndianUInt16(at: offset + 4)))
                guard x.isFinite, y.isFinite, z.isFinite else {
                    throw CacheError.nonFiniteVertex
                }
                vertices.append(SIMD3<Float>(x, y, z))
            }
            return vertices
        }
    }
}

private extension Data {
    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        var value = value.littleEndian
        Swift.withUnsafeBytes(of: &value) { append(contentsOf: $0) }
    }

    func littleEndianUInt16(at offset: Int) -> UInt16 {
        UInt16(self[offset]) | UInt16(self[offset + 1]) << 8
    }

    func littleEndianUInt32(at offset: Int) -> UInt32 {
        UInt32(self[offset])
            | UInt32(self[offset + 1]) << 8
            | UInt32(self[offset + 2]) << 16
            | UInt32(self[offset + 3]) << 24
    }
}
