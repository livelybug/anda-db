# Anda-LQL (Anda Logic Query Language) Specification v1.0 [DEPRECATED]

## Anda-LQL is now [KIP (Knowledge Interaction Protocol)](https://github.com/ldclabs/KIP).

## 1. Introduction

Anda-LQL is a declarative, graph-oriented query language designed for Large Language Models (LLMs), serving as the symbolic core of the Anda DB Cognitive Reasoning Engine (CRE). Its primary goal is to enable LLMs to retrieve information from Anda DB's "Cognitive Nexus" (a knowledge graph) safely, efficiently, and interpretably. Its syntax is designed to be LLM-friendly, breaking down complex intents into clear, logical blocks.

## 2. Core Concepts

*   **Knowledge Graph**: Composed of **Concept Nodes** and **Proposition Links**.
*   **Concept Node**: Represents an entity or concept (e.g., a drug, a disease). Each node has a unique `id`, a `type`, and multiple **Attributes** (e.g., `name`, `molecular_weight`, `risk_level`).
*   **Proposition Link**: Represents a fact in the form `(subject, predicate, object)`, connecting two concept nodes. E.g., `(Aspirin, treats, Headache)`. Abbreviated as `PROP`.
*   **Variable**: A placeholder in a query, prefixed with `?`. E.g., `?drug`.
*   **Literal**: A known value, such as a string enclosed in double quotes `"` or a number.

## 3. Query Structure

A complete Anda-LQL query consists of a series of clauses in the following order:

```prolog
FIND ...
WHERE {
  ...
}
ORDER BY ...
LIMIT N
OFFSET M
```

*   **`FIND`**: Declares the final output variables (including aggregations).
*   **`WHERE`**: Defines the core graph matching logic and constraints.
*   **`ORDER BY`**: Sorts the final result set.
*   **`LIMIT` / `OFFSET`**: Used for paginating results.

## 4. `FIND` Clause

Defines the query's output.

**Syntax:**
`FIND ?var1 (?agg_func(?var2) AS ?result_var) ...`

*   **Multi-variable Return**: Specifies one or more variables to return, e.g., `FIND ?drug ?symptom`.
*   **Aggregate Return**: Uses aggregation functions to perform calculations. An alias must be assigned using the `AS` keyword.

**Aggregation Functions:**

*   `COUNT(?var)`: Counts the number of bindings for `?var`. `COUNT(DISTINCT ?var)` counts unique bindings.
*   `COLLECT(?var)`: Collects all values of `?var` within a group into a list.

**Examples:**

```prolog
-- Return multiple variables
FIND ?drug_name ?symptom_name

-- Return a variable and its count
FIND ?drug_class (COUNT(?drug) AS ?drug_count)
```

## 5. `WHERE` Clause

Contains a series of graph pattern and filter clauses, implicitly joined by a logical **AND**.

### 5.1. Type Assertion / Entity Grounding Clause

**Function**: Constrains a variable's type or "grounds" it to a specific node.
**Syntax**: `?variable(type: "...", name: "...", id: "...")`
*   Parameters are optional, but at least one is required. This clause is used **only for constraining and locating nodes**.

**Examples**:
```prolog
?drug(type: "Drug")                -- Constrains ?drug to be of type "Drug"
?aspirin(name: "Aspirin")          -- Grounds ?aspirin to the node named "Aspirin"
?headache(id: "snomedct:25064002") -- Grounds ?headache to the node with a specific ID
```

### 5.2. Proposition Clause (`PROP`)

**Function**: Traverses the knowledge graph using a `(subject, predicate, object)` pattern.
**Syntax**: `PROP(Subject, Predicate, Object)`

**Example**:
```prolog
-- Find all drugs that treat Headache
PROP(?drug, "treats", ?headache)
```

### 5.3. Attribute Clause (`ATTR`)

**Function**: Retrieves an attribute's value from a node and binds it to a new variable.
**Syntax**: `ATTR(?node_variable, "attribute_name", ?value_variable)`

**Examples**:
```prolog
-- Get the "name" attribute of the ?drug node and bind it to ?drug_name
ATTR(?drug, "name", ?drug_name)

-- Get the "risk_level" attribute of the ?drug node and bind it to ?risk
ATTR(?drug, "risk_level", ?risk)
```

### 5.4. Filter Clause (`FILTER`)

**Function**: Applies complex conditions to bound variables (usually retrieved via `ATTR`).
**Syntax**: `FILTER(boolean_expression)`

**Filter Functions & Operators**:

*   **Comparison**: `==`, `!=`, `<`, `>`, `<=`, `>=`
*   **Logical**: `&&` (AND), `||` (OR), `!` (NOT)
*   **String**: `CONTAINS(?str, "sub")`, `STARTS_WITH(?str, "prefix")`, `ENDS_WITH(?str, "suffix")`, `REGEX(?str, "pattern")`
*   **Type Checks**: `isLiteral(?var)`, `isNumeric(?var)`

**Examples**:
```prolog
-- Filter for drugs with a risk level less than 3
ATTR(?drug, "risk_level", ?risk)
FILTER(?risk < 3)

-- Filter for drugs whose name contains "acid"
ATTR(?drug, "name", ?drug_name)
FILTER(CONTAINS(?drug_name, "acid"))
```

### 5.5. Negation Clause (`NOT`)

**Function**: Excludes solutions that match a specific pattern.
**Syntax**: `NOT { ... }`

**Example**:
```prolog
-- Exclude all drugs belonging to the NSAID class
NOT {
  ?nsaid_class(name: "NSAID")
  PROP(?drug, "is_class_of", ?nsaid_class)
}
```

### 5.6. Optional Clause (`OPTIONAL`)

**Function**: Tries to match an optional graph pattern. If the pattern matches, its variables are bound. If not, the query proceeds with those variables unbound. Similar to a SQL `LEFT JOIN`.
**Syntax**: `OPTIONAL { ... }`

**Example**:
```prolog
-- Find all drugs and, if available, their side effects
?drug(type: "Drug")
OPTIONAL {
  PROP(?drug, "has_side_effect", ?side_effect)
  ATTR(?side_effect, "name", ?side_effect_name)
}
```

### 5.7. Union Clause (`UNION`)

**Function**: Combines results from two or more logical blocks, implementing a logical **OR**.
**Syntax**: `{ ... } UNION { ... }`

**Example**:
```prolog
-- Find drugs that treat "Headache" or "Fever"
{
  ?headache(name: "Headache")
  PROP(?drug, "treats", ?headache)
}
UNION
{
  ?fever(name: "Fever")
  PROP(?drug, "treats", ?fever)
}
```

## 6. Solution Modifiers

These clauses post-process the result set generated by the `WHERE` clause.

*   **`ORDER BY ?var [ASC|DESC]`**: Sorts results by a variable. `ASC` (ascending) is the default.
*   **`LIMIT N`**: Limits the number of returned results to `N`.
*   **`OFFSET M`**: Skips the first `M` results, typically used with `LIMIT` for pagination.

## 7. Examples

#### Example 1: Advanced Query with Filtering and Sorting

**Intent**: "Find all non-NSAID drugs that treat 'Headache' with a risk level below 4. Return the drug name and risk level, sorted by risk level (ascending)."

```prolog
FIND ?drug_name ?risk
WHERE {
  ?drug(type: "Drug")
  ?headache(name: "Headache")
  ?nsaid_class(name: "NSAID")

  PROP(?drug, "treats", ?headache)

  NOT {
    PROP(?drug, "is_class_of", ?nsaid_class)
  }

  ATTR(?drug, "name", ?drug_name)
  ATTR(?drug, "risk_level", ?risk)
  FILTER(?risk < 4)
}
ORDER BY ASC(?risk)
LIMIT 20
```

#### Example 2: Analysis with Aggregation

**Intent**: "List the names of all drugs, grouped by their drug class."

```prolog
FIND ?class_name (COLLECT(?drug_name) AS ?drug_list)
WHERE {
  ?class(type: "DrugClass")
  ATTR(?class, "name", ?class_name)

  ?drug(type: "Drug")
  PROP(?drug, "is_class_of", ?class)
  ATTR(?drug, "name", ?drug_name)
}
ORDER BY ?class_name
```

#### Example 3: Handling Missing Information with `OPTIONAL`

**Intent**: "List all drugs in the NSAID class and, if available, their known side effects."

```prolog
FIND ?drug_name ?side_effect_name
WHERE {
  ?nsaid_class(name: "NSAID")
  PROP(?drug, "is_class_of", ?nsaid_class)
  ATTR(?drug, "name", ?drug_name)

  OPTIONAL {
    PROP(?drug, "has_side_effect", ?side_effect)
    ATTR(?side_effect, "name", ?side_effect_name)
  }
}
```
*   **Note**: For drugs with no side effects, `?side_effect_name` will be null, but the drug's `?drug_name` will still be included in the results.

## 8. LLM Generation Guide

1.  **Deconstruct Intent**: Break down the user's request into core goals, filter conditions, exclusions, and optional information.
2.  **Ground Entities**: Map named entities (e.g., "Aspirin") to graph nodes using `?var(name: "...")` or `?var(id: "...")`.
3.  **Build Relations**: Use `PROP` clauses to connect variables and build the core graph pattern.
4.  **Fetch Data**: Use `ATTR` to retrieve required node attributes (e.g., `name`, `risk_level`) and bind them to new variables.
5.  **Apply Filters**: Use `FILTER` to apply numerical or string constraints on variables retrieved via `ATTR`.
6.  **Handle Negation**: Use `NOT` to exclude unwanted patterns.
7.  **Handle Optionals**: If the request includes uncertain terms like "if any," use an `OPTIONAL` clause.
8.  **Define Output**: Declare the variables to return in the `FIND` clause. Use aggregation functions for statistics or lists.
9.  **Organize Results**: Use `ORDER BY` and `LIMIT` / `OFFSET` if sorting or pagination is needed.
