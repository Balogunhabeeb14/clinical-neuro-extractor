import pandas as pd
import pytest

from clinical_neuro_extractor.registry import build_patient_registry


@pytest.fixture
def documents() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "document_guid": "doc-1",
                "document_description": "report",
                "client_guid": "pt-1",
                "client_idcode": "MRN001",
                "client_touchedwhen": "2024-01-05",
                "clientvisit_admitdtm": "2024-01-01",
                "client_displayname": "Doe, Jane",
                "client_firstname": "Jane",
                "client_lastname": "Doe",
                "client_universalnumber": "SSN-1",
                "client_dob": "1980-04-12",
                "client_gendercode": "F",
                "client_racecode": "W",
                "client_title": "Ms",
                "client_applicsource": "intake",
                "body_analysed": "...",
                "updatetime": "2024-01-05",
                "clientvisit_visitidcode": "v1",
                "client_createdwhen": "2024-01-01",
                "clientvisit_typecode": "OP",
                "clientvisit_visitstatus": "closed",
                "clientvisit_internalvisitstatus": "closed",
                "clientvisit_carelevelcode": "outpatient",
                "clientvisit_serviceguid": "svc-1",
                "clientvisit_touchedby": "user1",
                "clientvisit_touchedwhen": "2024-01-05",
                "clientvisit_providerdisplayname_analysed": "Dr. Smith",
                "clientaddress_city": "Springfield",
                "clientaddress_postalcode": "62704",
                "client_languagecode": "EN",
                "client_religioncode": "NA",
            },
            {
                "document_guid": "doc-2",
                "document_description": "assessment",
                "client_guid": "pt-1",
                "client_idcode": "MRN001",
                "client_touchedwhen": "2024-03-10",
                "clientvisit_admitdtm": "2024-03-08",
                "client_displayname": "Doe, Jane",
                "client_firstname": "Jane",
                "client_lastname": "Doe",
                "client_universalnumber": "SSN-1",
                "client_dob": "1980-04-12",
                "client_gendercode": "F",
                "client_racecode": "W",
                "client_title": "Ms",
                "client_applicsource": "intake",
                "body_analysed": "...",
                "updatetime": "2024-03-10",
                "clientvisit_visitidcode": "v2",
                "client_createdwhen": "2024-03-08",
                "clientvisit_typecode": "OP",
                "clientvisit_visitstatus": "closed",
                "clientvisit_internalvisitstatus": "closed",
                "clientvisit_carelevelcode": "outpatient",
                "clientvisit_serviceguid": "svc-1",
                "clientvisit_touchedby": "user1",
                "clientvisit_touchedwhen": "2024-03-10",
                "clientvisit_providerdisplayname_analysed": "Dr. Jones",
                "clientaddress_city": "Springfield",
                "clientaddress_postalcode": "62704",
                "client_languagecode": "EN",
                "client_religioncode": "NA",
            },
            {
                "document_guid": "doc-3",
                "document_description": "report",
                "client_guid": "pt-2",
                "client_idcode": "MRN002",
                "client_touchedwhen": "2024-02-01",
                "clientvisit_admitdtm": "2024-01-30",
                "client_displayname": "Roe, Sam",
                "client_firstname": "Sam",
                "client_lastname": "Roe",
                "client_universalnumber": "SSN-2",
                "client_dob": "1990-07-20",
                "client_gendercode": "M",
                "client_racecode": "B",
                "client_title": "Mr",
                "client_applicsource": "referral",
                "body_analysed": "...",
                "updatetime": "2024-02-01",
                "clientvisit_visitidcode": "v3",
                "client_createdwhen": "2024-01-30",
                "clientvisit_typecode": "IP",
                "clientvisit_visitstatus": "closed",
                "clientvisit_internalvisitstatus": "closed",
                "clientvisit_carelevelcode": "inpatient",
                "clientvisit_serviceguid": "svc-2",
                "clientvisit_touchedby": "user2",
                "clientvisit_touchedwhen": "2024-02-01",
                "clientvisit_providerdisplayname_analysed": "Dr. Lee",
                "clientaddress_city": "Capital City",
                "clientaddress_postalcode": "62701",
                "client_languagecode": "EN",
                "client_religioncode": "NA",
            },
        ]
    )


def test_one_row_per_patient(documents):
    registry = build_patient_registry(documents)
    assert sorted(registry["client_guid"]) == ["pt-1", "pt-2"]
    assert len(registry) == 2


def test_rollup_counts_and_dates(documents):
    registry = build_patient_registry(documents).set_index("client_guid")

    pt1 = registry.loc["pt-1"]
    assert pt1["n_documents"] == 2
    assert pt1["n_visits"] == 2
    assert pt1["first_visit_dtm"] == pd.Timestamp("2024-01-01")
    assert pt1["last_visit_dtm"] == pd.Timestamp("2024-03-08")
    assert pt1["providers"] == ["Dr. Jones", "Dr. Smith"]


def test_report_and_assessment_counts(documents):
    registry = build_patient_registry(documents).set_index("client_guid")

    assert registry.loc["pt-1", "n_reports"] == 1
    assert registry.loc["pt-1", "n_assessments"] == 1
    assert registry.loc["pt-2", "n_reports"] == 1
    assert registry.loc["pt-2", "n_assessments"] == 0


def test_demographics_use_most_recently_touched_row(documents):
    docs = documents.copy()
    docs.loc[docs["document_guid"] == "doc-2", "clientaddress_city"] = "New City"

    registry = build_patient_registry(docs).set_index("client_guid")

    assert registry.loc["pt-1", "clientaddress_city"] == "New City"


def test_missing_client_guid_column_raises():
    with pytest.raises(KeyError):
        build_patient_registry(pd.DataFrame({"document_guid": ["doc-1"]}))


def test_deidentify_drops_direct_identifiers_and_hashes_id(documents):
    registry = build_patient_registry(documents, deidentify=True, salt="test-salt")

    for col in ("client_guid", "client_displayname", "client_firstname",
                "client_lastname", "client_universalnumber", "client_idcode",
                "client_dob"):
        assert col not in registry.columns

    assert "patient_id" in registry.columns
    assert "birth_year" in registry.columns
    assert set(registry["birth_year"]) == {1980, 1990}

    # postal code generalized to first 3 digits
    assert set(registry["clientaddress_postalcode"]) == {"627"}

    # hashing is deterministic for a given salt, and salt changes the hash
    registry_same_salt = build_patient_registry(documents, deidentify=True, salt="test-salt")
    assert list(registry["patient_id"]) == list(registry_same_salt["patient_id"])

    registry_diff_salt = build_patient_registry(documents, deidentify=True, salt="other-salt")
    assert list(registry["patient_id"]) != list(registry_diff_salt["patient_id"])
