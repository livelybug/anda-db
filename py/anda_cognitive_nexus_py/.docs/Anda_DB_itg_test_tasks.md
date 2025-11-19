# AndaDB Python FFI Integration - Task Breakdown

## Overview
This document lists the actionable tasks required to implement the Python FFI architecture for AndaDB, based on the technical design. Each task is prioritized, clearly defined, and includes dependencies and estimated effort.

---

## Task List

### 1. Project Setup & Dependencies
- **1.1.** Ensure `PyO3`, `pyo3-asyncio`, and `maturin` are added to `Cargo.toml` and project build system. **(Done)**
  - *Effort:* Low
  - *Dependencies:* None

### 2. Rust Core API Wrappers
- **2.1.** Refactor/create Rust functions to be FFI-friendly (async, error handling, type conversions).
  - *Effort:* Medium
  - *Dependencies:* 1.1
- **2.2.** Implement Rust struct/class to encapsulate DB state (e.g., `PyAndaDB` wrapping `Arc<CognitiveNexus>`).
  - *Effort:* Medium
  - *Dependencies:* 2.1
- **2.3.** Expose `create_kip_db` and `execute_kip` as #[pyfunction] and methods on the Python class.
  - *Effort:* Medium
  - *Dependencies:* 2.2
- **2.4.** Add docstrings to both methods for Python users.
  - *Effort:* Low
  - *Dependencies:* 2.3
- **2.5.** Add type hints in Python wrappers if possible.
  - *Effort:* Low
  - *Dependencies:* 2.3
- **2.6.** Add logging/tracing for debugging and performance (use Rust `log` crate).
  - *Effort:* Low
  - *Dependencies:* 2.3
- **2.7.** Refactor FFI result handling to return a tuple of (Python enum, Python dict) from `execute_kip`.
  - Expose the Rust `CommandType` enum as a Python enum using PyO3.
  - Convert the Rust `Response` to a Python dict using serde and PyO3 conversion traits.
  - Do not raise exceptions for errors; include error details in the returned dict.
  - Document the enum values and dict structure for Python users.
  - *Effort:* Medium
  - *Dependencies:* 2.3

### 3. Python API Design
- **3.1.** Design Python class interface (constructor, async methods, error mapping).
  - *Effort:* Low
  - *Dependencies:* 2.3
- **3.2.** Ensure all results are returned as Python dicts (serde conversion).
  - *Effort:* Low
  - *Dependencies:* 2.3

### 4. Async Runtime Integration
- **4.1.** Integrate pyo3-asyncio to bridge Rust Tokio runtime with Python asyncio event loop.
  - *Effort:* Medium
  - *Dependencies:* 2.3

### 5. Packaging & Distribution
- **5.1.** Design and finalize package name and metadata (authors, description, version, license, keywords, classifiers).
  - *Effort:* Low
  - *Dependencies:* 4.1
- **5.2.** Configure `maturin` and create/update `pyproject.toml` in `py/anda_cognitive_nexus_py` with correct metadata and [tool.maturin] section.
  - *Effort:* Low
  - *Dependencies:* 5.1
- **5.3.** Build Python wheel for multiple platforms using `maturin build --release` (test on Linux, document for macOS/Windows).
  - *Effort:* Low
  - *Dependencies:* 5.2
- **5.4.** Test wheel installation and import in a clean Python environment (venv).
  - *Effort:* Low
  - *Dependencies:* 5.3
- **5.5.** Prepare for PyPI publishing (`maturin publish`) and internal distribution (wheel file).
  - *Effort:* Low
  - *Dependencies:* 5.4
- **5.6.** Document installation, platform notes, and troubleshooting in README.md.
  - *Effort:* Low
  - *Dependencies:* 5.5

### 6. Testing & Validation
- **6.1.** Write Rust unit/integration tests for FFI wrappers and core logic.
  - *Effort:* Medium
  - *Dependencies:* 2.3
- **6.2.** Write Python integration tests (pytest) for the exposed API (class, async methods, error handling) using pytest-asyncio.
  - Place all Python integration tests in the directory `py/anda_cognitive_nexus_py/tests_py`.
  - Use pytest-asyncio for all test functions to cover async usage only.
  - Cover PyAndaDB.create and PyAndaDB.execute_kip, including both success and failure cases.
  - Validate returned data structure (tuple, JSON string) and type conversions.
  - Do not include synchronous wrapper tests.
  - *Effort:* Medium
  - *Dependencies:* 5.2
- **6.3.** Test and document behavior when db_config contains non-JSON-serializable Python objects.
  - *Effort:* Low
  - *Dependencies:* 2.3
- **6.4.** Test and document strict type validation and error messages for AndaDbConfig.
  - *Effort:* Low
  - *Dependencies:* 2.3
- **6.5.** Test thread-safety and async compatibility of PyAndaDB in Python.
  - *Effort:* Low
  - *Dependencies:* 2.3
- **6.6.** Test Python-side unpacking and ergonomics of returned tuples; consider returning Python dict directly.
  - *Effort:* Low
  - *Dependencies:* 2.3
- **6.7.** Test error messages for invalid command/parameters and improve user-friendliness.
  - *Effort:* Low
  - *Dependencies:* 2.3
- **6.8.** Test error handling for non-JSON-serializable input.
  - *Effort:* Low
  - *Dependencies:* 2.3
- **6.9.** Test and evaluate if returning a Python dict directly from execute_kip improves usability.
  - *Effort:* Low
  - *Dependencies:* 2.3

### 7. Documentation
- **7.1.** Document Python API usage, installation, and example workflows.
  - *Effort:* Low
  - *Dependencies:* 6.2
- **7.2.** Update Rust doc comments for FFI-exposed functions and classes.
  - *Effort:* Low
  - *Dependencies:* 2.3

---

## Task 2.1 Planned Changes: Refactor/Create Rust Functions for FFI

- Ensure all public Rust functions intended for Python FFI are `async` and compatible with PyO3/pyo3-asyncio.
- Refactor error handling so that errors are returned as `BoxError` and can be mapped to Python exceptions generically.
- Update function signatures and struct fields to use Python-friendly types (e.g., serde_json::Value, Python dicts via serde, snake_case fields).
- Remove or wrap complex Rust types (Arc, trait objects) so they can be safely passed across the FFI boundary.
- Add doc comments and PyO3 attributes (e.g., #[pyfunction], #[pymethods]) to functions and structs intended for Python exposure.
- Ensure all results are serializable to Python dicts using serde.
- Prepare for integration with a Python class API (to be implemented in later tasks).
- Validate changes with Rust unit tests to ensure FFI compatibility and correct error propagation.

## Prioritization Summary
1. Project setup & dependencies
2. Rust FFI wrappers and class
3. Python API design
4. Async runtime integration
5. Packaging
6. Testing
7. Documentation

## Notes
- Tasks are sequential where dependencies exist; parallelization possible for documentation and testing after core API is ready.
- Estimated effort: Low (1-2h), Medium (2-6h), High (6h+)