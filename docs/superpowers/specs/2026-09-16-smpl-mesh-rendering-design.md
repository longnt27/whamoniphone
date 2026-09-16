# SMPL Mesh Rendering Design

## Goal

Render the selected mobile WHAM pipeline's complete SMPL body mesh on iPhone without changing model inference or inflating the existing JSON result format.

The app already executes `WHAM_WorldStep` and receives one world-space tensor with 6,890 vertices for every inferred frame. The analyzer currently discards that tensor. This feature persists those vertices efficiently and teaches the existing SceneKit views to display them as a shaded surface.

## User Experience

- A successfully analyzed video opens with a blue SMPL surface rather than only the 17-joint skeleton.
- The video view and free-camera 3D view both offer a Mesh/Skeleton control.
- Mesh is the default when both mesh frames and topology are available.
- Skeleton remains available for inspection and as the automatic fallback.
- Existing analyses that predate the mesh cache continue to open as skeletons.
- A missing, corrupt, or incompatible cache never prevents the JSON result from opening.

This renderer visualizes the world-space result. It does not claim pixel-accurate perspective registration over the source video; that would require camera intrinsics and projection that the current overlay does not retain.

## Data Flow

`WhamAnalyzer` continues to run the selected pipeline unchanged:

```text
YOLO26m-pose -> HMR2-S -> token adapter -> WHAM_I
             -> WHAM_ImageStep -> smoothing -> WHAM_WorldStep
```

For each `WHAM_WorldStep` result:

1. The existing pose, shape, translation, and joint fields go to JSON.
2. `vertices_world` goes to a streaming mesh-cache writer.
3. The first inferred mesh is written twice so cache frame zero matches the JSON warm-start duplicate and every JSON index maps directly to the same mesh index.
4. The completed cache is atomically published beside the JSON only after all frames succeed.

The sidecar URL derives from the JSON output URL by replacing `.json` with `.whammesh`. The JSON schema stays backward-compatible and does not contain thousands of vertex numbers.

## Mesh Cache Format

The binary format is versioned and little-endian:

| Field | Type | Value |
| --- | --- | --- |
| Magic | 8 bytes | `WHAMSMPL` |
| Version | UInt32 | `1` |
| Vertex count | UInt32 | `6890` |
| Frame count | UInt32 | finalized atomically |
| Components | UInt32 | `3` |
| Payload | Float16 | frame-major XYZ triples |

Each frame occupies 41,340 bytes. At 30 FPS the cache grows by about 74 MB per minute. Float16 preserves substantially more precision than the display needs while halving Float32 storage. The reader memory-maps the file and decodes only the requested frame into SceneKit vertices.

The reader rejects:

- an incorrect magic value or version;
- any vertex/component count other than 6,890 by 3;
- a declared length that does not exactly match the file length;
- an out-of-range frame index; or
- non-finite decoded coordinates.

The writer uses a temporary sibling file. Failure or cancellation leaves the previous complete cache untouched, and a successful finalization replaces it.

## SMPL Topology and Licensing

SMPL connectivity is fixed at 13,776 triangles. The user's existing licensed HMR checkpoint contains it as `smpl.faces_tensor`.

A repository script extracts that tensor, validates that it is an integer array shaped `[13776, 3]`, validates every index is in `0..<6890`, and writes `SMPLFaces.bin` using UInt16 indices plus a small versioned header. The generated binary is ignored by Git because it is derived from a licensed model. The extraction script, format description, and setup command are committed.

The application loads `SMPLFaces.bin` as an optional bundle resource. A clone without the licensed resource still builds and runs, but displays the skeleton until its owner generates the topology file.

No topology is downloaded from an unofficial mirror, embedded in source code, or added to repository history.

## Rendering Architecture

`SMPLMeshCache` owns the binary writer and memory-mapped reader. It has no UI responsibilities.

`SMPLTopology` owns topology decoding and validation. Production loads it from the application bundle; tests provide small in-memory fixtures.

`SMPLMeshGeometry` converts one decoded frame plus the fixed face list into SceneKit geometry. It computes smooth per-vertex normals from triangle cross-products, uses a double-sided blue material, and applies the same `(x, -y, -z)` coordinate conversion as the existing skeleton.

The existing SceneKit engine gains one mesh node and presentation-mode control. It keeps the skeleton nodes so switching modes does not reconstruct the scene. Updating a frame replaces only the mesh geometry and joint/bone transforms; the camera and lights remain stable.

`AnalyzedTabView` loads the JSON and, independently, attempts to open the matching mesh cache. Both child views receive the optional reader. They default to Mesh only when topology and the cache are valid, otherwise to Skeleton. Playback avoids decoding the same mesh index more than once.

## Failure Handling

- If inference fails, neither a partial JSON nor a partial mesh cache is published.
- If topology extraction receives the wrong checkpoint or tensor shape, it exits nonzero and does not write an output.
- If the bundle topology is absent, the app shows skeleton mode without crashing.
- If the cache is corrupt or has fewer frames than JSON, the affected view falls back to skeleton mode and exposes a short status label.
- Selecting Mesh while it is unavailable is disabled rather than producing an empty SceneKit view.

## Compatibility and Scope

The inference graph, model selection, smoothing parameters, evaluation metrics, and benchmark path do not change. The renderer consumes the already-computed `vertices_world` output.

This feature does not add textures, gender-specific models, multi-person rendering, perspective-calibrated video compositing, mesh export, or live-camera rendering. Those are separate follow-ups.

The existing uncommitted changes in `OfflineProcessView.swift` and `utils/convert_pkl.py` are not part of this work and must remain untouched.

## Verification

Implementation follows test-driven development. Automated tests cover:

- binary header and Float16 frame round-trip;
- exact frame-count and warm-start alignment;
- bad magic, unsupported version, truncated payload, invalid counts, and out-of-range reads;
- topology header, triangle count, and vertex-index validation;
- hand-derived vertex normals on a small triangle fixture; and
- mesh-unavailable fallback selection.

After unit tests pass, verification includes:

1. all existing `WhamAppTests` on an iPhone simulator;
2. a Release device build with code signing;
3. confirmation that the local app bundle contains `SMPLFaces.bin` when generated;
4. a short on-device analysis showing the mesh and Skeleton/Mesh switch; and
5. inspection that JSON remains lightweight and the `.whammesh` frame count equals the JSON frame count.

