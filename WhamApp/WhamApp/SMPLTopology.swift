import Foundation

struct SMPLTopology: Sendable {
    static let expectedVertexCount = 6_890
    static let expectedTriangleCount = 13_776

    private static let magic = Array("SMPLFACE".utf8)
    private static let version: UInt32 = 1
    private static let indicesPerTriangle: UInt32 = 3
    private static let headerSize = 24

    enum TopologyError: LocalizedError {
        case invalidLength
        case invalidMagic
        case unsupportedVersion(UInt32)
        case unexpectedVertexCount(expected: Int, actual: Int)
        case unexpectedTriangleCount(expected: Int, actual: Int)
        case invalidIndicesPerTriangle(UInt32)
        case vertexIndexOutOfRange(UInt16)

        var errorDescription: String? {
            switch self {
            case .invalidLength:
                return "Invalid SMPL topology file length"
            case .invalidMagic:
                return "Not an SMPL topology file"
            case .unsupportedVersion(let version):
                return "Unsupported SMPL topology version: \(version)"
            case .unexpectedVertexCount(let expected, let actual):
                return "Expected \(expected) SMPL vertices, topology contains \(actual)"
            case .unexpectedTriangleCount(let expected, let actual):
                return "Expected \(expected) SMPL triangles, topology contains \(actual)"
            case .invalidIndicesPerTriangle(let count):
                return "SMPL topology contains \(count) indices per triangle"
            case .vertexIndexOutOfRange(let index):
                return "SMPL topology vertex index is out of range: \(index)"
            }
        }
    }

    let vertexCount: Int
    let triangleCount: Int
    let indices: [UInt16]

    init(
        data: Data,
        expectedVertexCount: Int = SMPLTopology.expectedVertexCount,
        expectedTriangleCount: Int = SMPLTopology.expectedTriangleCount
    ) throws {
        guard data.count >= Self.headerSize else {
            throw TopologyError.invalidLength
        }
        guard Array(data.prefix(Self.magic.count)) == Self.magic else {
            throw TopologyError.invalidMagic
        }
        let version = data.smplUInt32(at: 8)
        guard version == Self.version else {
            throw TopologyError.unsupportedVersion(version)
        }
        let vertexCount = Int(data.smplUInt32(at: 12))
        guard vertexCount == expectedVertexCount else {
            throw TopologyError.unexpectedVertexCount(
                expected: expectedVertexCount,
                actual: vertexCount
            )
        }
        let triangleCount = Int(data.smplUInt32(at: 16))
        guard triangleCount == expectedTriangleCount else {
            throw TopologyError.unexpectedTriangleCount(
                expected: expectedTriangleCount,
                actual: triangleCount
            )
        }
        let indexCount = data.smplUInt32(at: 20)
        guard indexCount == Self.indicesPerTriangle else {
            throw TopologyError.invalidIndicesPerTriangle(indexCount)
        }

        let (totalIndices, indexOverflow) = triangleCount.multipliedReportingOverflow(by: 3)
        let (payloadBytes, payloadOverflow) = totalIndices.multipliedReportingOverflow(by: 2)
        let (expectedLength, lengthOverflow) = Self.headerSize.addingReportingOverflow(payloadBytes)
        guard !indexOverflow,
              !payloadOverflow,
              !lengthOverflow,
              expectedLength == data.count else {
            throw TopologyError.invalidLength
        }

        var indices: [UInt16] = []
        indices.reserveCapacity(totalIndices)
        for index in 0..<totalIndices {
            let value = data.smplUInt16(at: Self.headerSize + index * 2)
            guard value < UInt16(vertexCount) else {
                throw TopologyError.vertexIndexOutOfRange(value)
            }
            indices.append(value)
        }

        self.vertexCount = vertexCount
        self.triangleCount = triangleCount
        self.indices = indices
    }

    static func loadFromBundle(_ bundle: Bundle = .main) -> SMPLTopology? {
        guard let url = bundle.url(forResource: "SMPLFaces", withExtension: "bin"),
              let data = try? Data(contentsOf: url) else {
            return nil
        }
        return try? SMPLTopology(data: data)
    }
}

private extension Data {
    func smplUInt16(at offset: Int) -> UInt16 {
        UInt16(self[offset]) | UInt16(self[offset + 1]) << 8
    }

    func smplUInt32(at offset: Int) -> UInt32 {
        UInt32(self[offset])
            | UInt32(self[offset + 1]) << 8
            | UInt32(self[offset + 2]) << 16
            | UInt32(self[offset + 3]) << 24
    }
}
