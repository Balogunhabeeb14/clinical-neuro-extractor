# clinical-neuro-extractor

## Patient registry

`clinical_neuro_extractor.build_patient_registry` collapses a document/visit-level
dataframe of neuropsychological patients (one row per clinical document,
repeated per patient) into a one-row-per-patient registry with rolled-up
visit/document stats.

Expected input columns (`clinical_neuro_extractor.DOCUMENT_FIELDS`):

```
document_guid client_guid client_idcode client_touchedwhen clientvisit_admitdtm
client_displayname client_firstname client_lastname client_universalnumber
client_dob client_gendercode client_racecode client_title client_applicsource
document_description body_analysed updatetime clientvisit_visitidcode
client_createdwhen clientvisit_typecode clientvisit_visitstatus
clientvisit_internalvisitstatus clientvisit_carelevelcode clientvisit_serviceguid
clientvisit_touchedby clientvisit_touchedwhen clientvisit_providerdisplayname_analysed
clientaddress_city clientaddress_postalcode client_languagecode client_religioncode
```

### Usage

```python
import pandas as pd
from clinical_neuro_extractor import build_patient_registry

documents = pd.read_csv("documents.csv")  # or however your dataframe is built

registry = build_patient_registry(documents)
```

Each row of `registry` is one patient (`client_guid`), with demographics taken
from their most recently touched document plus:

- `n_documents`, `n_visits`, `n_reports`, `n_assessments`
- `first_visit_dtm`, `last_visit_dtm`, `first_document_dtm`, `last_touched_dtm`
- `visit_types`, `care_levels`, `providers` (sorted lists of distinct values seen)

To produce a de-identified registry (hashed patient ID, no name/DOB/MRN/SSN,
birth year instead of DOB, 3-digit postal code prefix):

```python
registry = build_patient_registry(documents, deidentify=True, salt="a-private-per-project-salt")
```

## Document content: reports vs. assessments

`document_description` is either `"report"` (a free-text narrative) or
`"assessment"` (a battery/domain/test/subtest score sheet), and `body_analysed`
holds that document's free text.

### Report sections

Reports have a reliable heading structure (REASON FOR REFERRAL, BACKGROUND
INFORMATION, PRESENTATION, ASSESSMENT FINDINGS - itself split into cognitive
domain subsections like MEMORY FUNCTIONS - CONCLUSIONS, ...).
`clinical_neuro_extractor.parse_report_sections` splits one report's
`body_analysed` into a `(section, subsection, text)` dataframe;
`parse_report_sections_batch` runs it over every `document_description ==
"report"` row in a documents dataframe and tags each section with
`document_guid`/`client_guid`.

```python
from clinical_neuro_extractor import parse_report_sections_batch

sections = parse_report_sections_batch(documents)
```

### Assessment scores

Assessment score sheets do **not** have a reliable structural pattern -
they're free text pulled out of a PDF table where labels and their values
often land on separate, irregularly-ordered lines, so a rule-based/regex
parser can't reliably reconstruct battery/domain/test/subtest/value
structure. `clinical_neuro_extractor.assessment_llm_extractor` instead uses
Claude to read the text and extract structured rows. It requires the `llm`
extra (`pip install -e ".[llm]"`):

```python
from clinical_neuro_extractor.assessment_llm_extractor import extract_assessment_measures_batch

measures = extract_assessment_measures_batch(documents)
# columns: document_guid, client_guid, battery, domain, test, subtest, metric, value
```

A document that fails to extract doesn't abort the batch - its row carries
an `error` column instead, so nothing silently disappears. This is a
best-effort extraction (the model's read of messy free text), not a
guaranteed-accurate structured parse - spot-check results before relying on
them clinically.

### Report clinical/demographic details

Beyond the section split above, `clinical_neuro_extractor.report_llm_extractor`
pulls clinically useful facts out of a report's prose using the same
Claude + Pydantic approach: diagnosis, laterality, treatment type, whether
the assessment was pre- or post-treatment, referral reason, and
demographic context (occupation, handedness, education, marital status,
living situation) mentioned in the text - plus each cognitive domain's
overall rating (e.g. Memory: high average) pulled from ASSESSMENT
FINDINGS/CONCLUSIONS. Many reports don't mention treatment at all (e.g. a
pre-surgical baseline) - a null `treatment_type`/`treatment_timing` is a
normal, correct result, not a failure. Also requires the `llm` extra.

```python
from clinical_neuro_extractor.report_llm_extractor import extract_report_details_batch

details, domain_summaries = extract_report_details_batch(documents)
# details columns: document_guid, client_guid, diagnosis, diagnosis_laterality,
#   treatment_type, treatment_timing, treatment_details,
#   reason_for_referral_summary, occupation, handedness, education,
#   marital_status, living_situation, other_demographic_notes, error
# domain_summaries columns: document_guid, client_guid, domain, overall_level, notes
```

Note: structured demographics you already have as columns in the source
dataframe (name, DOB, gender, race, address, language, religion) don't need
extraction - this module is for demographic/clinical detail only mentioned
in the report's free text (occupation, handedness, etc.), not a replacement
for those fields.

### Development

```bash
pip install -e ".[dev,llm]"
pytest
```
