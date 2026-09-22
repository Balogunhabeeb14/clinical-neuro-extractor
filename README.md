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

- `n_documents`, `n_visits`
- `first_visit_dtm`, `last_visit_dtm`, `first_document_dtm`, `last_touched_dtm`
- `visit_types`, `care_levels`, `providers` (sorted lists of distinct values seen)

To produce a de-identified registry (hashed patient ID, no name/DOB/MRN/SSN,
birth year instead of DOB, 3-digit postal code prefix):

```python
registry = build_patient_registry(documents, deidentify=True, salt="a-private-per-project-salt")
```

### Development

```bash
pip install -e ".[dev]"
pytest
```
