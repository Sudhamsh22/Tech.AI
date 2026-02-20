from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .schema import SupportDocument

ISSUE_LABELS = [
    "connectivity",
    "hardware_failure",
    "software_error",
    "performance",
    "power_battery",
]

SEED_TRAINING_EXAMPLES: List[Tuple[str, str]] = [
    ("Laptop cannot connect to office wifi, keeps dropping network", "connectivity"),
    ("Bluetooth mouse is not detected after reboot", "connectivity"),
    ("No display on startup and fan spins loudly", "hardware_failure"),
    ("Keyboard keys not responding on business notebook", "hardware_failure"),
    ("Blue screen appears after latest windows update", "software_error"),
    ("Driver installation fails with error code 43", "software_error"),
    ("Laptop is very slow when opening browser and excel", "performance"),
    ("System freezes when many apps are open", "performance"),
    ("Battery drains from 80 to 20 in one hour", "power_battery"),
    ("Device not charging even when adapter is connected", "power_battery"),
]


def sample_knowledge_base() -> Sequence[SupportDocument]:
    return [
        SupportDocument(
            doc_id="KB-001",
            title="Wi-Fi Troubleshooting for Enterprise Laptops",
            product_family="business",
            source="internal_faq",
            content=(
                "If Wi-Fi disconnects, first toggle airplane mode, then update wireless driver, "
                "reset TCP/IP stack, and verify VPN profile is not forcing invalid DNS settings."
            ),
        ),
        SupportDocument(
            doc_id="KB-002",
            title="Battery Health and Charging Diagnostics",
            product_family="consumer",
            source="service_manual",
            content=(
                "For charging failures inspect adapter wattage, DC jack seating, and battery health report. "
                "Run BIOS diagnostics for battery cycle count and AC adapter detection."
            ),
        ),
        SupportDocument(
            doc_id="KB-003",
            title="Blue Screen Recovery Procedure",
            product_family="gaming",
            source="os_guide",
            content=(
                "For blue screen errors collect stop code, boot safe mode, remove recent drivers, "
                "and run sfc /scannow plus memory diagnostics."
            ),
        ),
        SupportDocument(
            doc_id="KB-004",
            title="Performance Degradation Runbook",
            product_family="business",
            source="ticket_playbook",
            content=(
                "When laptops become slow, inspect startup applications, check disk utilization, "
                "scan for malware, and ensure thermal throttling is not occurring due to blocked vents."
            ),
        ),
    ]


def label_to_index() -> Dict[str, int]:
    return {label: i for i, label in enumerate(ISSUE_LABELS)}
