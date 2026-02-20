from tech_support_ai.assistant import TechSupportAssistantModel


def main() -> None:
    assistant = TechSupportAssistantModel.build_default()
    print("Tech Support Assistant (type 'exit' to quit)")

    while True:
        query = input("\nIssue> ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        response = assistant.answer(query)
        print(f"\nCategory: {response.issue_category} ({response.confidence:.2f})")
        print(f"Answer: {response.answer}")
        print("Steps:")
        for idx, step in enumerate(response.steps, start=1):
            print(f"  {idx}. {step}")
        print(f"Evidence: {', '.join(response.evidence_doc_ids)}")


if __name__ == "__main__":
    main()
