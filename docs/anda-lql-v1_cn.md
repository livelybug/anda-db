# Anda-LQL (Anda Logic Query Language) 规范 v1.0

## 1. 简介

Anda-LQL 是一种为大型语言模型 (LLM) 设计的声明式、图导向的查询语言，作为 Anda DB“思维推理引擎”(CRE) 的符号核心。其主要目标是让 LLM 从 Anda DB 的“认知中枢”（一个知识图谱）中安全、高效、可解释地检索信息。它的语法被设计成对 LLM 生成友好，将复杂的意图分解为清晰的逻辑块。

## 2. 核心概念

*   **知识图谱 (Knowledge Graph)**: 由**概念节点 (Concept Nodes)** 和 **命题链接 (Proposition Links)** 构成。
*   **概念节点 (Concept Node)**: 代表实体或概念（如药物、疾病）。每个节点都拥有一个唯一的 `id` 和一个 `type`，以及多个**属性 (Attributes)**，如 `name`, `molecular_weight`, `risk_level` 等。
*   **命题链接 (Proposition Link)**: 代表一个事实，形式为 `(主语, 谓词, 宾语)`，连接两个概念节点，如：`(阿司匹林, treats, 头痛)`，简写为 `PROP`。
*   **变量 (Variable)**: 以 `?` 开头，如 `?drug`。是查询中的占位符。
*   **字面量 (Literal)**: 用双引号 `"` 括起来的字符串或数字，代表已知值。

## 3. 查询结构

一个完整的 Anda-LQL 查询由一系列子句构成，遵循以下顺序：

```prolog
FIND ...
WHERE {
  ...
}
ORDER BY ...
LIMIT N
OFFSET M
```

*   **`FIND`**: 声明最终输出的变量（包括聚合结果）。
*   **`WHERE`**: 定义图谱匹配的核心逻辑和约束。
*   **`ORDER BY`**: 对最终结果集进行排序。
*   **`LIMIT` / `OFFSET`**: 用于对结果进行分页。

## 4. `FIND` 子句

定义查询的输出。v2.0 进行了大幅增强。

**语法:**
`FIND ?var1 (?agg_func(?var2) AS ?result_var) ...`

*   **多变量返回**: 可以指定一个或多个变量，如 `FIND ?drug ?symptom`。
*   **聚合返回**: 可以使用聚合函数（见第 7 节）对变量进行计算。必须使用 `AS` 关键字为聚合结果指定一个新的变量名。

**聚合函数 (Aggregation Functions):**

*   `COUNT(?var)`: 计算 `?var` 被绑定的次数。`COUNT(DISTINCT ?var)` 计算不同绑定的数量。
*   `COLLECT(?var)`: 将一个分组内 `?var` 的所有值收集成一个列表。

**示例:**

```prolog
-- 返回多个变量
FIND ?drug_name ?symptom_name

-- 返回一个变量和它的计数值
FIND ?drug_class (COUNT(?drug) AS ?drug_count)
```

## 5. `WHERE` 子句

包含一系列图模式匹配和过滤子句，所有子句之间默认为逻辑 **AND** 关系。

### 5.1. 类型断言/实体接地子句

**功能**: 约束变量类型或将变量“接地”到图谱中的特定节点。
**语法**: `?variable(type: "...", name: "...", id: "...")`
*   参数可选，但至少提供一个。此子句**只用于约束和定位**。

**示例**:
```prolog
?drug(type: "Drug")              -- 约束 ?drug 必须是药物类型
?aspirin(name: "Aspirin")         -- 将 ?aspirin 接地到名为 "Aspirin" 的节点
?headache(id: "snomedct:25064002") -- 将 ?headache 接地到指定 ID 的节点
```

### 5.2. 命题子句 (`PROP`)

**功能**: 在知识图谱中按 `(主语, 谓词, 宾语)` 模式进行遍历。
**语法**: `PROP(Subject, Predicate, Object)`

**示例**:
```prolog
-- 找到所有能治疗头痛的药物
PROP(?drug, "treats", ?headache)
```

### 5.3. 属性子句 (`ATTR`)

**功能**: 获取一个节点的属性值，并将其绑定到一个新变量上。
**语法**: `ATTR(?node_variable, "attribute_name", ?value_variable)`

**示例**:
```prolog
-- 获取 ?drug 节点的 "name" 属性，并存入 ?drug_name 变量
ATTR(?drug, "name", ?drug_name)

-- 获取 ?drug 节点的 "risk_level" 属性，并存入 ?risk 变量
ATTR(?drug, "risk_level", ?risk)
```

### 5.4. 过滤器子句 (`FILTER`)

**功能**: 对已绑定的变量（通常是 `ATTR` 获取的属性值）应用更复杂的过滤条件。
**语法**: `FILTER(boolean_expression)`

**过滤器函数与运算符 (Filter Functions & Operators)**:

*   **比较运算符**: `==`, `!=`, `<`, `>`, `<=`, `>=`
*   **逻辑运算符**: `&&` (AND), `||` (OR), `!` (NOT)
*   **字符串函数**: `CONTAINS(?str, "sub")`, `STARTS_WITH(?str, "prefix")`, `ENDS_WITH(?str, "suffix")`, `REGEX(?str, "pattern")`
*   **类型检查**: `isLiteral(?var)`, `isNumeric(?var)`

**示例**:
```prolog
-- 筛选出风险等级小于 3 的药物
ATTR(?drug, "risk_level", ?risk)
FILTER(?risk < 3)

-- 筛选出名称包含 "acid" 的药物
ATTR(?drug, "name", ?drug_name)
FILTER(CONTAINS(?drug_name, "acid"))
```

### 5.5. 否定子句 (`NOT`)

**功能**: 排除满足特定模式的解。
**语法**: `NOT { ... }`

**示例**:
```prolog
-- 排除所有属于 NSAID 类的药物
NOT {
  ?nsaid_class(name: "NSAID")
  PROP(?drug, "is_class_of", ?nsaid_class)
}
```

### 5.6. 可选子句 (`OPTIONAL`)

**功能**: 尝试匹配一个可选的图模式。如果模式匹配成功，则其内部变量被绑定；如果失败，查询继续，但内部变量为未绑定状态。这类似于 SQL 的 `LEFT JOIN`。
**语法**: `OPTIONAL { ... }`

**示例**:
```prolog
-- 查找所有药物，并（如果存在的话）一并找出它们的副作用
?drug(type: "Drug")
OPTIONAL {
  PROP(?drug, "has_side_effect", ?side_effect)
  ATTR(?side_effect, "name", ?side_effect_name)
}
```

### 5.7. 合并子句 (`UNION`)

**功能**: 合并两个或多个独立逻辑块的结果，实现逻辑 **OR**。
**语法**: `{ ... } UNION { ... }`

**示例**:
```prolog
-- 找到能治疗“头痛”或“发烧”的药物
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

## 6. Solution Modifiers (结果修饰子句)

这些子句在 `WHERE` 逻辑执行完毕后，对产生的结果集进行后处理。

*   **`ORDER BY ?var [ASC|DESC]`**:
    根据指定变量对结果进行排序，默认为 `ASC`（升序）。
*   **`LIMIT N`**:
    限制返回结果的数量为 N。
*   **`OFFSET M`**:
    跳过前 M 条结果，通常与 `LIMIT` 联用实现分页。

## 7. 综合查询示例

#### 示例 1: 带过滤和排序的高级查询

**意图**: "找到所有能治疗‘头痛’的非 NSAID 类药物，要求其风险等级低于4，并按风险等级从低到高排序，返回药物名称和风险等级。"

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

#### 示例 2: 使用聚合分析查询

**意图**: "按药物类别，列出该类别下所有药物的名称。"

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

#### 示例 3: 使用 `OPTIONAL` 处理缺失信息

**意图**: "列出所有 NSAID 类的药物，并（如果存在的话）显示它们各自的已知副作用。"

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
*   **注意**: 对于没有副作用的药物，`?side_effect_name` 的值将为空，但药物本身 `?drug_name` 依然会出现在结果中。

## 8. LLM 生成指南

1.  **分解意图 (Deconstruct)**: 将用户请求分解为核心查询目标、过滤条件、排除条件和可选信息。
2.  **接地实体 (Ground)**: 使用 `?var(name: "...")` 或 `?var(id: "...")` 将请求中的具名实体（“阿司匹林”）映射到图谱节点。
3.  **构建关系 (Relate)**: 使用 `PROP` 子句连接变量，构建核心的图查询模式。
4.  **获取数据 (Attribute)**: 明确需要哪些节点的属性值（如 `name`, `risk_level`），使用 `ATTR` 将它们绑定到新变量。
5.  **施加过滤 (Filter)**: 对通过 `ATTR` 获取的属性变量，使用 `FILTER` 施加数值或字符串约束。
6.  **处理否定 (Negate)**: 使用 `NOT` 子句排除不希望出现的情况。
7.  **处理可选 (Optional)**: 如果请求中包含“如果有的话”、“可能存在”等不确定信息，使用 `OPTIONAL` 子句进行匹配。
8.  **定义输出 (Find)**: 根据最终目标，在 `FIND` 中声明要返回的变量。如果需要统计或列表，使用聚合函数。
9.  **组织结果 (Organize)**: 如果需要排序或分页，使用 `ORDER BY` 和 `LIMIT` / `OFFSET`。
