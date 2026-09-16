# SMPL Mesh Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the 6,890 SMPL vertices already emitted by the mobile WHAM pipeline and render them as a shaded mesh in the iPhone app.

**Architecture:** Stream world-space vertices to a versioned Float16 sidecar aligned one-to-one with JSON frames, memory-map that sidecar during playback, and combine each frame with licensed fixed SMPL topology loaded from an optional bundle resource. Extend the existing SceneKit engine so mesh is the default when available and skeleton remains a selectable fallback.

**Tech Stack:** Swift 6, Core ML, SceneKit, SwiftUI, Swift Testing, Python/PyTorch asset extraction, Xcode 16+

**Spec:** `docs/superpowers/specs/2026-09-16-smpl-mesh-rendering-design.md`

## Global Constraints

- Deployment remains iOS 18.0 or newer.
- The selected inference pipeline and smoothing behavior must not change.
- The 13,776-face licensed SMPL topology must remain outside Git history.
- Existing JSON-only analyses must continue to render as skeletons.
- `WhamApp/WhamApp/OfflineProcessView.swift` and `utils/convert_pkl.py` contain unrelated user edits and must not be modified or committed.
- Production code follows red-green-refactor; every new behavioral unit begins with a failing test.

---

### Task 1: Versioned SMPL mesh cache

**Files:**
- Create: `WhamApp/WhamApp/SMPLMeshCache.swift`
- Modify: `WhamApp/WhamAppTests/WhamAppTests.swift`

**Interfaces:**
- Produces: `SMPLMeshCache.Writer.init(outputURL:vertexCount:)`, `append(_:)`, `finalize()` and `SMPLMeshCache.Reader.init(url:)`, `frame(at:)`.
- Produces: `SMPLMeshCache.outputURL(forJSONURL:)` so analyzer and viewer derive the same sidecar path.

- [ ] **Step 1: Write failing cache tests**

Add tests that write two hand-authored three-vertex frames to a temporary URL, finalize the writer, read both frames, and compare literal coordinates within Float16 tolerance. Add separate tests for bad magic, truncation, an out-of-range frame, and sidecar URL derivation.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
xcodebuild test -project WhamApp/WhamApp.xcodeproj -scheme WhamApp -destination 'platform=iOS Simulator,name=iPhone 17' -only-testing:WhamAppTests/WhamAppTests
```

Expected: compilation fails because `SMPLMeshCache` does not exist.

- [ ] **Step 3: Implement the streaming format**

Implement the 24-byte `WHAMSMPL` header, little-endian UInt32 fields, Float16 XYZ payload, strict length/count validation, finite-value checks, sibling temporary file, and atomic finalization. `frame(at:)` returns `[SIMD3<Float>]` and decodes only one requested frame.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: new cache tests and all existing unit tests pass.

- [ ] **Step 5: Commit**

```bash
git add WhamApp/WhamApp/SMPLMeshCache.swift WhamApp/WhamAppTests/WhamAppTests.swift
git commit -m "feat: add versioned SMPL mesh cache"
```

### Task 2: Licensed topology extraction and validation

**Files:**
- Create: `tools/export/export_smpl_topology.py`
- Create: `WhamApp/WhamApp/SMPLTopology.swift`
- Modify: `.gitignore`
- Modify: `WhamApp/WhamAppTests/WhamAppTests.swift`
- Generate locally, never add: `WhamApp/WhamApp/SMPLFaces.bin`

**Interfaces:**
- Produces: Python CLI `--checkpoint PATH --output PATH` that extracts `smpl.faces_tensor`.
- Produces: `SMPLTopology.init(data:expectedVertexCount:)` and `SMPLTopology.loadFromBundle()`.
- Binary contract: `SMPLFACE` magic, version 1, UInt32 vertex/triangle counts, then UInt16 triangle indices.

- [ ] **Step 1: Write failing topology decoder tests**

Create a literal one-triangle binary fixture and assert indices `[0, 1, 2]`. Add independent failures for a wrong magic value and an index equal to the vertex count.

- [ ] **Step 2: Run tests and verify RED**

Run the Task 1 test command. Expected: compilation fails because `SMPLTopology` does not exist.

- [ ] **Step 3: Implement decoder and bundle fallback**

Validate header fields, exact payload length, triangle count, and every index. Bundle loading returns `nil` when the licensed resource is absent or invalid so existing clones keep working.

- [ ] **Step 4: Implement and exercise the extractor**

Load the checkpoint with PyTorch, accept either top-level or `state_dict` payloads, find a key ending in `smpl.faces_tensor`, validate shape `[13776, 3]` and index range `0..<6890`, and atomically write the binary resource. Generate `SMPLFaces.bin` from the existing downloaded checkpoint and confirm it is ignored by Git.

- [ ] **Step 5: Run tests and verify GREEN**

Run the Task 1 test command. Expected: all topology and existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add .gitignore tools/export/export_smpl_topology.py WhamApp/WhamApp/SMPLTopology.swift WhamApp/WhamAppTests/WhamAppTests.swift
git commit -m "feat: load licensed SMPL mesh topology"
```

### Task 3: Shaded SceneKit mesh geometry

**Files:**
- Create: `WhamApp/WhamApp/SMPLMeshGeometry.swift`
- Modify: `WhamApp/WhamApp/Skeleton3DEngine.swift`
- Modify: `WhamApp/WhamAppTests/WhamAppTests.swift`

**Interfaces:**
- Produces: `SMPLMeshGeometry.vertexNormals(vertices:faces:) -> [SIMD3<Float>]`.
- Produces: `SMPLMeshGeometry.make(vertices:topology:) -> SCNGeometry`.
- Produces: `Skeleton3DEngine.meshAvailable`, `presentationMode`, and `applyFrameData(keypoints3D:meshVertices:)`.

- [ ] **Step 1: Write failing normal and fallback tests**

For the triangle `(0,0,0)`, `(1,0,0)`, `(0,1,0)`, assert all three normals are exactly `(0,0,1)`. Assert an engine without topology selects skeleton presentation even when mesh is requested.

- [ ] **Step 2: Run tests and verify RED**

Run the focused unit-test command. Expected: compilation fails because `SMPLMeshGeometry` and mesh presentation do not exist.

- [ ] **Step 3: Implement geometry and engine integration**

Build vertex and normal geometry sources plus a UInt16 triangle element. Use a double-sided blue physically based material. Add one persistent mesh node to the existing scene; hide skeleton nodes in mesh mode and hide the mesh node in skeleton mode. Apply `(x, -y, -z)` consistently to mesh and joints.

- [ ] **Step 4: Run tests and verify GREEN**

Run the focused unit-test command. Expected: all geometry, fallback, and existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add WhamApp/WhamApp/SMPLMeshGeometry.swift WhamApp/WhamApp/Skeleton3DEngine.swift WhamApp/WhamAppTests/WhamAppTests.swift
git commit -m "feat: render shaded SMPL geometry"
```

### Task 4: Persist inference vertices and expose mesh UI

**Files:**
- Modify: `WhamApp/WhamApp/WhamAnalyzer.swift`
- Modify: `WhamApp/WhamApp/VideoLibraryView.swift`
- Modify: `WhamApp/WhamApp/VideoDetailView.swift`
- Modify: `WhamApp/WhamAppTests/WhamAppTests.swift`

**Interfaces:**
- Consumes: `SMPLMeshCache`, `SMPLTopology`, and mesh-capable `Skeleton3DEngine`.
- Produces: `VideoModel.smplMeshURL` and an analyzed viewer that defaults to mesh when cache and topology are valid.

- [ ] **Step 1: Write failing alignment and mode-selection tests**

Extract a small pure helper that defines cache write indices for the first inferred frame and assert source frame one emits cache indices zero and one while every later source frame emits one aligned index. Test that valid cache plus topology selects mesh and any missing dependency selects skeleton.

- [ ] **Step 2: Run tests and verify RED**

Run the focused unit-test command. Expected: new alignment or selection APIs are absent.

- [ ] **Step 3: Stream vertices during analysis**

Create the writer before recurrent inference, append the first `verticesWorld` twice, append later frames once, verify final cache count equals final JSON count, finalize the cache before atomically publishing JSON, and discard the temporary cache on errors.

- [ ] **Step 4: Integrate playback and controls**

Load the optional memory-mapped reader beside JSON. Pass it to both views, add a Mesh/Skeleton segmented control, default according to availability, disable Mesh when unavailable, decode a frame only when its index changes, and display a concise fallback label for invalid or legacy results.

- [ ] **Step 5: Run tests and verify GREEN**

Run the focused test command. Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add WhamApp/WhamApp/WhamAnalyzer.swift WhamApp/WhamApp/VideoLibraryView.swift WhamApp/WhamApp/VideoDetailView.swift WhamApp/WhamAppTests/WhamAppTests.swift
git commit -m "feat: display WHAM SMPL mesh results"
```

### Task 5: Documentation and full verification

**Files:**
- Modify: `README.md`
- Modify: `docs/FINAL_REPORT.md`

**Interfaces:**
- Documents the generated licensed resource, cache storage cost, fallback behavior, and renderer scope.

- [ ] **Step 1: Document topology generation and rendering**

Add the exact extractor command, explain that `SMPLFaces.bin` is local and licensed, record the `.whammesh` size of about 74 MB/minute at 30 FPS, and replace the report's “renderer is follow-up” statement with the implemented behavior and projection limitation.

- [ ] **Step 2: Run full verification**

Run all simulator unit tests, then build Release for the connected iPhone destination with signing. Inspect the built `.app` to verify `SMPLFaces.bin` is present and confirm `git status` does not show the binary.

- [ ] **Step 3: Review the complete diff**

Run `git diff --check`, inspect every changed file, and confirm the two unrelated dirty files remain outside all feature commits.

- [ ] **Step 4: Commit and push**

```bash
git add README.md docs/FINAL_REPORT.md
git commit -m "docs: explain on-device SMPL rendering"
git push origin main
```

