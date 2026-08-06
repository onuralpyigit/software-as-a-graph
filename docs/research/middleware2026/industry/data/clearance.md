# Publication clearance and anonymization record

Fill this in before §5 is finalized. Its content drives two things in the paper: the anonymization
sentence in §5.1 and the acknowledgement/disclosure footnote.

## Clearance

- **Granting body / approver:** `TODO(data)`
- **Date of approval:** `TODO(data)`
- **Reference (ticket, memo, email):** `TODO(data)`
- **Scope of approval:** `TODO(data)` — which of {system scale, audit latency, defect counts,
  incident counts} may be published, and at what granularity.

## Anonymization applied

Record each transformation, because §5.1 must state it:

| Item | Treatment | Rationale |
|---|---|---|
| Program / platform name | `TODO(data)` (e.g. "referred to as Platform A") | |
| Absolute entity counts | `TODO(data)` (exact / rounded to nearest 10 / order of magnitude) | |
| Defect categories | `TODO(data)` (published as-is / generalized) | |
| Incident counts | `TODO(data)` (absolute / normalized per release) | |
| Vendor and middleware product names | `TODO(data)` | |

## Draft sentence for §5.1

> The deployment is reported under the anonymization required by its operator: `TODO(data)`.
> Figures are `TODO(data: exact | rounded)`; relative changes are reported alongside absolute
> counts wherever clearance permits.

## Author disclosure

- **Industry-affiliated author(s):** `TODO(data)` — the Middleware Industrial Track requires at
  least one.
- **Conflict / funding statement:** `TODO(data)`
