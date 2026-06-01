# Analyzer Module — CodeQL Mode Rules

## Modes

| Mode | Flag | Description |
|------|------|-------------|
| `manual` | `--analysis-mode manual` | Existing method: XML parse + import parse. Default. |
| `codeql` | `--analysis-mode codeql` | CodeQL call graph analysis. |

## CodeQL Mode

### Purpose

Starting from the `main` method, detect **write** (pub) and **read** (sub) calls reachable through any call chain. Extract only the topics that are **actually reached**, not merely declared.

### Reachability Scope

**All** of the following are included in the transitive call chain starting from `main`:

- Direct method calls
- Virtual dispatch (polymorphism)
- Interface dispatch
- Lambda / method reference
- Anonymous inner class
- Callback / event listener
- Reflection (`Method.invoke`)
- Nested / chained calls

### Topic Name Extraction

The topic name is not a string literal. It is extracted via regex from the **class type** of a specific argument.

Example:
```
custom_write(abcInstance)
  → abcInstance type: Abc_class
  → regex: "(.*)_class"
  → topic name: "Abc"
```

- The argument index (`topic_arg_index`) is specific to each write/read method definition.
- The topic name regex (`topic_name_pattern`) is global.

### Output Rules

#### pub / sub

Every write call reachable from `main` → **application's pub**.
Every read call reachable from `main` → **application's sub**.

Write/read calls reached through library code are also the **application's** pub/sub. No distinction is made between "direct" and "via library".

```csv
App_X,TopicA,pub
App_X,TopicB,sub
```

#### uses

If a **library's code** is traversed in the call chain leading to a write/read call → `uses` relationship.

```csv
App_X,MyLib,uses
```

- Importing alone is **not sufficient**. The library code must be **actually entered** in the call chain.
- Not imported but entered library code via transitive call → uses **applies**.

#### Library Self-Analysis

Libraries also have a `main` method. When a library is analyzed independently, the write/read calls reachable from its own `main` are extracted:

```csv
MyLib,TopicA,pub
```

### Output Format

**Identical** to the existing CSV format:

- File name: `{versioned_name}.csv`
- Header row: **None**
- Separator: `,`
- Columns: `entity_name,target,role`
- Role values: `pub`, `sub`, `uses`

### Configuration (runtime.yaml)

```yaml
analyzer:
  # Existing fields (for manual mode)
  import_domain_prefix: "a.b.c"
  dependency_suffixes:
    - "_lib"
  dummy_topic_names:
    - "dummytopic"
  makefile_include_patterns:
    - "include/Makefile.mk"

  # CodeQL configuration
  codeql:
    cli_path: "/usr/local/bin/codeql"
    main_method_name: "main"
    topic_name_pattern: "(.*)_class"
    write_methods:
      - class_name: "WriterA"
        method_name: "custom_write"
        topic_arg_index: 0
      - class_name: "WriterB"
        method_name: "do_write"
        topic_arg_index: 1
    read_methods:
      - class_name: "ReaderA"
        method_name: "custom_read"
        topic_arg_index: 0
```

| Key | Description |
|-----|-------------|
| `cli_path` | Path to the CodeQL CLI binary |
| `main_method_name` | Root method name for the call graph |
| `topic_name_pattern` | Regex (with capture group) to extract topic name from class name |
| `write_methods` | Method definitions that produce pub relationships (multiple allowed) |
| `read_methods` | Method definitions that produce sub relationships (multiple allowed) |
| `class_name` | Class name where the write/read call is made |
| `method_name` | Write/read method name |
| `topic_arg_index` | Argument index containing the topic type (0-based) |

### Manual Mode

- Existing logic works as-is: XML parse (`xml_parser.py`) + import parse (`java_parser.py`).
- CodeQL configuration is not used.
- `java_parser.py` is not invoked in CodeQL mode.

### Mode Comparison

| Feature | Manual | CodeQL |
|---------|--------|--------|
| Topic source | XML `<topic>` | Call graph write/read calls |
| Uses source | Java import parse | Library code traversed in call chain |
| Evidence level | Declarative (what is defined) | Reachability (what is actually called) |
| Library inclusion | Uses if imported | Uses if library code is in call chain |
| Output format | Same CSV | Same CSV |
