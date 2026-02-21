from tech_support_ai.assistant import TechSupportAssistantModel


def test_answer_returns_structured_response() -> None:
    assistant = TechSupportAssistantModel.build_default()
    response = assistant.answer("my laptop keeps disconnecting from wifi")

    assert response.issue_category in {
        "connectivity",
        "hardware_failure",
        "software_error",
        "performance",
        "power_battery",
    }
    assert 0.0 <= response.confidence <= 1.0
    assert len(response.steps) >= 3
    assert len(response.evidence_doc_ids) > 0
