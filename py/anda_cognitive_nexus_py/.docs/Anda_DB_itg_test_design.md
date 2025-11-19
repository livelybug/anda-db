# `execute_kip` Function - Implementation and Test Design

This document outlines the design for implementing and testing a new `execute_kip` function in the `anda_cognitive_nexus_py` crate. This plan supersedes previous versions.

### 1. Goal

The primary goal is to create a single, reusable Rust function, `execute_kip`, that can execute KIP (Knowledge Interaction Protocol) commands. This function will be tested using two runnable integration tests located in the `examples/` directory.

### 2. Architectural Design

#### 2.1. Core Function Implementation

*   **Crate:** `anda_cognitive_nexus_py`
*   **Location:** `tools/anda_cognitive_nexus_py/src/lib.rs`
*   **Function:** `execute_kip`
*   **Signature:**
    ```rust
    pub async fn execute_kip(
        command: String,
        parameters: serde_json::Value,
        dry_run: bool
    ) -> Result<(anda_kip::CommandType, anda_kip::Response), anda_core::BoxError>
    ```                                                                     
* Logic:
    1. The function will substitute the placeholders (e.g., $name) in the command string with values
       from the parameters JSON object.
    2. It will establish a connection to an in-memory AndaDB and instantiate the CognitiveNexus.
    3. It will parse the substituted command string.
    4. It will use the CognitiveNexus to execute the parsed command.
    5. It will return the (CommandType, Response) tuple on success or an error if any step fails.

#### 2.2. Integration Test Implementation

*   **Location for New Files:** `tools/anda_cognitive_nexus_py/examples/`
*   **Proposed Filenames:**
    1.  `test_kip_stateful_execution.rs`
    2.  `test_kip_validation.rs`

### **3. Component Interfaces and Interactions**                                                   
                                                                                                   
1.  **`main` function (`#[tokio::main]`)**: Entry point for each example.                          
2.  **Test Environment Setup**:                                                                    
    *   Instantiate `AndaDB` in memory mode.                                                       
    *   Instantiate `CognitiveNexus` using `CognitiveNexus::connect()`, passing it the             
ndaDB` instance.                                                                                   
3.  **`CognitiveNexus::execute` (Function Under Test)**:                                           
    *   **Interface:** `async fn execute(&self, command: Command, dry_run: bool) ->                
sponse`                                                                                            
    *   **Interaction:** The test will parse a KIP command string into a `Command`                 
ject and pass it to this function with the relevant `dry_run` flag.                                
4.  **Assertions**:                                                                                
    *   **`test_kip_validation.rs`**:                                                              
        1.  Call `nexus.execute` with a `CREATE CONCEPT` command and `dry_run=True`.               
        2.  Assert the result is `Ok`.                                                             
        3.  Call `nexus.execute` again with a `MATCH CONCEPT` command.                             
        4.  Assert that the `MATCH` result is empty, proving no data was written.                  
    *   **`test_kip_stateful.rs`**:                                                                
        1.  Call `nexus.execute` with a `CREATE CONCEPT` command and `dry_run=False`.              
        2.  Assert the result is `Ok`.                                                             
        3.  Call `nexus.execute` again with a `MATCH CONCEPT` command.                             
        4.  Assert the `MATCH` result contains the newly created concept.                          
                                                                                                   
---                                                                                                
                                                                                                   
### **4. Visual Diagrams**                                                                         
                                                                                                   
#### **4.1. Flowchart: Test Execution Process**                                                    
```mermaid                                                                                         
flowchart TD                                                                                       
    A[Start] --> B{Create Tokio Runtime};                                                          
    B --> C[Instantiate In-Memory AndaDB];                                                         
    C --> D[Instantiate CognitiveNexus];                                                           
    D --> E[Create KIP Request (command, dry_run flag)];                                           
    E --> F[Call CognitiveNexus.execute];                                                          
    F --> G{Check dry_run flag};                                                                   
    G -- dry_run = true --> I[Assert CREATE was Ok];                                               
    I --> J[Call CognitiveNexus.execute with MATCH];                                               
    J --> K[Assert MATCH result is empty];                                                         
    K --> Z[End];                                                                                  
    G -- dry_run = false --> L[Assert CREATE was Ok];                                              
    L --> M[Call CognitiveNexus.execute with MATCH];                                               
    M --> N[Assert MATCH result contains created data];                                            
    N --> Z;                                                                                       
```                                                                                                
                                                                                                   
#### **4.2. Sequence Diagram: Stateful Test (`dry_run=False`)**                                    
```mermaid                                                                                         
sequenceDiagram                                                                                    
    participant Test as Test Runner                                                                
    participant Runtime as Tokio Runtime                                                           
    participant Nexus as CognitiveNexus                                                            
    participant DB as AndaDB                                                                       
                                                                                                   
    Test->>Runtime: block_on(async_main)                                                           
    Runtime->>Nexus: connect()                                                                     
    Nexus->>DB: new(DBMode::Memory)                                                                
    DB-->>Nexus: db_instance                                                                       
    Nexus-->>Runtime: nexus_instance                                                               
    Runtime-->>Test: nexus_instance                                                                
                                                                                                   
    Test->>Nexus: execute(CREATE, dry_run=false)                                                   
    Nexus->>DB: Write or No-Op (based on dry_run)                                                  
    DB-->>Nexus: Result                                                                            
    Nexus-->>Test: Response                                                                        
                                                                                                   
    Test->>Test: assert!(result is correct)                                                        
```
### 5. Dependencies

The `anda_cognitive_nexus` crate has been added to the `[dependencies]` section of `tools/anda_cognitive_nexus_py/Cargo.toml` as it is required by the library code. No other dependencies are required for the tests that are not already available.

## 6. Python FFI Architecture & Technical Design

### 6.1. FFI Technology Choice
- Use [PyO3](https://pyo3.rs/) and [pyo3-asyncio](https://docs.rs/pyo3-asyncio/) to expose Rust async functions to Python.
- Package as a Python wheel using [maturin](https://github.com/PyO3/maturin) for easy distribution and installation.

### 6.2. API Design
- Expose a Python class (e.g., `AndaDB`) that encapsulates the DB state (CognitiveNexus, AndaDB instance).
- Provide async methods (e.g., `execute_kip`) on the class for KIP command execution.
- The class is instantiated via a Rust-backed constructor (wrapping `create_kip_db`).
- All results are returned as Python dicts for ergonomic Python usage.
- Errors are mapped to a generic Python exception for now.

### 6.3. Data Types & Interop
- Complex Rust types (Arc, trait objects, custom structs/enums) are wrapped or converted to Python-friendly types.
- Use serde for conversion between Rust and Python dicts.
- Python users interact with the API using native Python types (dicts, strings, etc.).

### 6.4. Async Support
- All exposed methods are async and can be awaited from Python (using asyncio).
- The Rust async runtime (Tokio) is bridged to Python's event loop via pyo3-asyncio.

### 6.5. Testing & Integration
- Integration tests are runnable from Python (pytest), using the exposed Python API.
- Rust unit/integration tests remain for core logic validation.

### 6.6. Summary of Flow
1. Python user instantiates `AndaDB` (calls Rust `create_kip_db`).
2. User calls `await anda_db.execute_kip(...)` with command, parameters, and dry_run flag.
3. Rust substitutes parameters, executes command, and returns result as Python dict.
4. Errors are raised as generic Python exceptions.
