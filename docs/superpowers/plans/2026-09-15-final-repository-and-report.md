# Final Repository and Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the validated mobile WHAM implementation into focused shared components, organize reproducible evidence and historical experiments, and publish an evidence-backed final report and operational README.

**Architecture:** Keep video I/O and benchmark orchestration separate while sharing model names, tensor types, preprocessing, and causal recurrent/world execution. Preserve all selected model behavior and retain raw evidence verbatim; move superseded experiments into a labelled archive instead of deleting them.

**Tech Stack:** Swift 5, SwiftUI, Core ML, Core Image, AVFoundation, XCTest/Swift Testing, Python 3 evaluation utilities, Markdown, JSON.

**Spec:** `docs/superpowers/specs/2026-09-15-final-repository-and-report-design.md`

## Global Constraints

- The selected pipeline remains YOLO26m-pose → released HMR2.0-S → selected residual token adapter → WHAM_I → WHAM_ImageStep → output-only light smoothing → WHAM_WorldStep.
- Preserve float16 recurrent/world tensors, float32 WHAM initializer tensors, alpha 0.75 pose smoothing, alpha 0.35 shape smoothing, and Ultralytics-compatible 640-pixel letterboxing.
- Do not feed smoothed pose back into WHAM recurrent state.
- Do not alter or trim source measurement JSON.
- Keep superseded experiment source in `archive/experiments`; remove only caches and generated/recoverable clutter.
- Commit each independently verifiable task.

---

### Task 1: Commit the validated pipeline baseline

**Files:**
- Modify: `.gitignore`
- Modify: `WhamApp/WhamApp/PoseDetector.swift`
- Modify: `WhamApp/WhamApp/WhamAnalyzer.swift`
- Modify: `WhamApp/WhamApp/WhamApp.swift`
- Create: `WhamApp/WhamApp/BenchmarkView.swift`
- Create: `WhamApp/WhamApp/WhamBenchmark.swift`
- Create: `WhamApp/WhamApp/TemporalOutputSmoother.swift`
- Create: `WhamApp/WhamApp/BenchmarkFrames/*.jpg`
- Test: `WhamApp/WhamAppTests/WhamAppTests.swift`
- Create: `utils/export_hmr2s_token_adapter.py`

**Interfaces:**
- Consumes: the seven ignored `.mlpackage` directories in `WhamApp/WhamApp`.
- Produces: the validated Release-build application and schema-version-8 benchmark JSON.

- [ ] **Step 1: Run the focused unit tests**

Run:

```bash
xcodebuild test -project WhamApp/WhamApp.xcodeproj -scheme WhamApp \
  -configuration Debug -destination 'platform=iOS Simulator,name=iPhone 17' \
  -only-testing:WhamAppTests CODE_SIGNING_ALLOWED=NO
```

Expected: all YOLO parser and temporal-smoothing tests pass.

- [ ] **Step 2: Build the physical-device Release target**

Run:

```bash
xcodebuild -project WhamApp/WhamApp.xcodeproj -scheme WhamApp \
  -configuration Release -destination 'id=00008030-00016C1E2668802E' \
  -derivedDataPath /tmp/wham-device-derived build
```

Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 3: Commit only the selected pipeline baseline**

```bash
git add .gitignore WhamApp/WhamApp/PoseDetector.swift \
  WhamApp/WhamApp/WhamAnalyzer.swift WhamApp/WhamApp/WhamApp.swift \
  WhamApp/WhamApp/BenchmarkView.swift WhamApp/WhamApp/WhamBenchmark.swift \
  WhamApp/WhamApp/TemporalOutputSmoother.swift \
  WhamApp/WhamApp/BenchmarkFrames WhamApp/WhamAppTests/WhamAppTests.swift \
  utils/export_hmr2s_token_adapter.py
git commit -m "feat: implement selected mobile WHAM pipeline"
```

### Task 2: Extract shared mobile pipeline components

**Files:**
- Create: `WhamApp/WhamApp/MobileWhamModelCatalog.swift`
- Create: `WhamApp/WhamApp/MobileWhamTypes.swift`
- Create: `WhamApp/WhamApp/MobileWhamPreprocessor.swift`
- Create: `WhamApp/WhamApp/MobileWhamCore.swift`
- Modify: `WhamApp/WhamApp/WhamAnalyzer.swift`
- Modify: `WhamApp/WhamApp/WhamBenchmark.swift`
- Test: `WhamApp/WhamAppTests/WhamAppTests.swift`

**Interfaces:**
- Consumes: `PoseDetector.Detection`, selected Core ML model outputs, gyro tensors.
- Produces: `MobileWhamObservation`, `MobileWhamFrontendOutput`, `MobileWhamCore.State`, `initialize(...)`, and `step(...)` shared by analyzer and benchmark.

- [ ] **Step 1: Add failing catalog and preprocessing contract tests**

Add tests asserting the exact seven model resource names, the selected hashes,
and normalized missing-observation shapes/dtypes. Build to observe failures
because the shared types do not yet exist.

- [ ] **Step 2: Implement the model catalog, data types, and preprocessor**

Move the existing constants and preprocessing code without altering formulas,
thresholds, crop scale, dtypes, or output shapes.

- [ ] **Step 3: Run focused tests**

Run the Task 1 unit-test command. Expected: all tests pass.

- [ ] **Step 4: Add a failing output-only recurrent-state test**

Add a test around a small state-update helper showing that raw WHAM pose becomes
`previousPose` while smoothed pose is returned only for rendering/world decode.
Run it and observe the missing shared core failure.

- [ ] **Step 5: Implement `MobileWhamCore` and replace duplicate analyzer/benchmark state code**

Extract first-frame initialization and one recurrent/world step. Accept loaded
`MLModel` instances so the analyzer can retain its two-phase memory strategy.
Expose an optional timing closure for the benchmark without placing benchmark
logic in the production core.

- [ ] **Step 6: Run unit tests, simulator build, and Release build**

Expected: tests pass and both builds succeed.

- [ ] **Step 7: Commit the refactor**

```bash
git add WhamApp/WhamApp/MobileWham*.swift \
  WhamApp/WhamApp/WhamAnalyzer.swift WhamApp/WhamApp/WhamBenchmark.swift \
  WhamApp/WhamAppTests/WhamAppTests.swift
git commit -m "refactor: share mobile WHAM pipeline components"
```

### Task 3: Organize current tooling, evidence, and archive

**Files:**
- Create: `tools/export/*`
- Create: `tools/evaluation/*`
- Create: `tools/kaggle/*`
- Create: `archive/experiments/*`
- Create: `evaluation/results/selected/*`
- Create: `evaluation/results/selected/manifest.json`
- Delete: generated Repomix outputs and cache directories from the working tree

**Interfaces:**
- Consumes: current `utils` scripts/notebooks and downloaded result archives.
- Produces: canonical current reproduction paths and immutable selected evidence.

- [ ] **Step 1: Write a manifest validator that initially fails**

Create `tools/evaluation/validate_final_evidence.py` to require the grid,
smoothing, device, and export reports; verify SHA-256 values and cross-report
selection fields. Run it before copying evidence and observe the expected
missing-file failure.

- [ ] **Step 2: Add verbatim selected evidence and its manifest**

Store canonical reports below `evaluation/results/selected` and include source
archive/file names plus hashes. The device JSON must be byte-for-byte identical
to the downloaded report.

- [ ] **Step 3: Move current scripts and notebooks into focused tool folders**

Keep only the selected HMR2-S/YOLO grid, smoothing, adapter export, HMR2-S
export, WHAM split/world export, their notebook builders, and dependency files
under `tools`. Update imports and notebook builder paths mechanically.

- [ ] **Step 4: Move superseded FastViT/BEDLAM/deployment experiments into the archive**

Preserve source and notebooks below `archive/experiments` with a README that
states they are historical and not part of the selected pipeline.

- [ ] **Step 5: Run the validator and Python syntax checks**

```bash
python3 tools/evaluation/validate_final_evidence.py
python3 -m compileall -q tools archive/experiments
```

Expected: validation succeeds and all Python source compiles.

- [ ] **Step 6: Commit repository organization and evidence**

```bash
git add tools archive evaluation/results/selected .gitignore
git add -u utils repomix-output.xml
git commit -m "chore: organize evaluation evidence and research archive"
```

### Task 4: Write the final report and operational README

**Files:**
- Create: `docs/FINAL_REPORT.md`
- Rename: `readme.md` → `README.md`
- Modify: `evaluation/README.md`

**Interfaces:**
- Consumes: validated selected evidence and manifest.
- Produces: one scientific narrative and one concise operator entry point.

- [ ] **Step 1: Add documentation checks to the evidence validator**

Require the final report to mention dataset/population, all four accuracy
metrics, device, median latency, cold loading, memory, thermal state, the YOLO
stall, limitations, and exact evidence links. Require the README to contain the
selected pipeline, build/run steps, model list, and reproduction commands. Run
the validator and observe the expected missing-document failure.

- [ ] **Step 2: Write `docs/FINAL_REPORT.md`**

Tell the chronological experiment story, define each metric in human terms,
separate selection from locked testing, compare released/mobile accuracy,
analyze smoothing, interpret device latency without hiding the stall, state
limitations, and give a bounded conclusion.

- [ ] **Step 3: Rewrite and case-normalize `README.md`**

Lead with the outcome, list the exact selected architecture, summarize measured
accuracy/latency, document model placement, build/run/export steps, link the
report/evidence, and identify archived experiments.

- [ ] **Step 4: Replace the stale evaluation index**

Make `evaluation/README.md` a short map of selected evidence, methodology, and
historical reports; remove claims that FastViT is the deployable path.

- [ ] **Step 5: Run evidence/document checks and link validation**

```bash
python3 tools/evaluation/validate_final_evidence.py
rg -n 'FastViTNormalized|YOLOv8n-pose.*current|neutral/zero fallback' README.md docs/FINAL_REPORT.md evaluation/README.md
```

Expected: validator passes; stale-current terminology search returns no match.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md docs/FINAL_REPORT.md evaluation/README.md
git add -u readme.md
git commit -m "docs: publish final mobile WHAM evaluation story"
```

### Task 5: Final verification

**Files:**
- Modify only if verification exposes a defect.

**Interfaces:**
- Consumes: completed refactor, evidence, and documentation.
- Produces: fresh test/build/validation evidence and clean focused diffs.

- [ ] **Step 1: Run the full Swift unit suite**

Use the Task 1 unit-test command. Expected: zero failures.

- [ ] **Step 2: Run a fresh signed Release build**

Use the Task 1 Release-build command with a fresh derived-data directory.
Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 3: Run Python evidence, syntax, whitespace, and stale-path checks**

```bash
python3 tools/evaluation/validate_final_evidence.py
python3 -m compileall -q tools archive/experiments
git diff --check HEAD~4..HEAD
rg -n 'utils/' README.md docs/FINAL_REPORT.md evaluation/README.md tools archive
```

Expected: validators pass; remaining `utils/` matches exist only inside
historical notebook content and are explicitly documented as historical.

- [ ] **Step 4: Review commit boundaries and repository status**

Confirm each significant change has its own commit and list any intentionally
uncommitted pre-existing user files rather than including them accidentally.
