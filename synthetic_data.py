from __future__ import annotations

import random

from config import synthetic_source_name, synthetic_source_url


# simple source material for template filling
# These lists give the templates some variation so the synthetic messages are not all identical.

delivery_companies = ["USPS", "UPS", "FedEx", "DHL"]
bank_names = ["RiverBank", "Pioneer Credit", "Union Trust", "North Valley Bank"]
government_sources = ["DMV", "county court", "state toll service", "tax office"]
prize_brands = ["Target", "Costco", "Amazon", "Walmart"]
person_names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley"]

amount_values = ["$4.99", "$12.80", "$35.00", "$89.20", "$250.00", "$1,000.00"]
point_values = ["480", "900", "1,250", "2,400", "3,000"]
case_numbers = ["C-1842", "T-5501", "A-9033", "P-7710", "R-4418", "V-6624"]
reference_codes = ["REF284", "REF510", "REF773", "REF812", "REF945", "REF991"]
tracking_codes = ["TRK3812", "TRK5921", "TRK6640", "TRK7425", "TRK8804", "TRK9317"]

time_windows = ["before 2 PM", "within 1 hour", "today", "before midnight", "within 30 minutes"]

example_links = [
    "https://example.com/verify",
    "https://example.com/claim",
    "https://example.com/pay",
    "https://example.com/update",
]


# scam templates

scam_templates = {
    "impersonation": [
        "{name}, this is your manager. Buy 3 gift cards {time_window} and send the codes here. Ref {reference_code}.",
        "Notice from {government_source}: your case {case_number} is pending review. Confirm your identity at {link}.",
        "Your {bank_name} fraud team needs you to verify recent login {reference_code} now at {link}.",
    ],
    "urgency": [
        "Urgent: your account will be locked {time_window} unless you confirm your details at {link}.",
        "Final reminder: act now to avoid service interruption on case {case_number}. Update your billing profile at {link}.",
        "Immediate action needed. Your benefits expire {time_window} unless you submit verification now.",
    ],
    "threat": [
        "Final notice from {government_source}: unpaid balance recorded under {case_number}. Failure to respond may trigger enforcement action.",
        "Your driving privileges will be suspended if you do not settle violation {case_number} immediately at {link}.",
        "Court alert: missing payment for case {case_number} may result in additional penalties and legal review.",
    ],
    "payment_request": [
        "You owe an overdue fee of {amount}. Pay now to avoid added charges on {reference_code}: {link}.",
        "Invoice due today for your account balance of {amount}. Submit payment for {reference_code} at {link}.",
        "A service charge of {amount} is pending. Clear balance {reference_code} now to prevent a hold.",
    ],
    "prize_scam": [
        "Congratulations! You won a {brand} gift card. Claim prize {reference_code} now at {link}.",
        "You were selected for a cash bonus of {amount}. Confirm reward {reference_code} {time_window}.",
        "Your loyalty account has {points} expiring points. Redeem batch {reference_code} now at {link}.",
    ],
    "account_verification": [
        "Security alert: verify your {bank_name} profile now to stop unauthorized access on {reference_code}.",
        "We detected unusual activity on your account. Confirm login {reference_code} immediately at {link}.",
        "Your payment app needs quick account verification for transfer batch {reference_code} before service can continue.",
    ],
    "delivery_scam": [
        "{delivery_company}: package {tracking_code} is on hold due to unpaid postage of {amount}. Pay at {link}.",
        "{delivery_company} delivery failed for shipment {tracking_code}. Update shipping preferences now using {link}.",
        "Parcel notice: shipment {tracking_code} is waiting for address confirmation before final delivery.",
    ],
}


# helper

def build_generation_context(random_generator: random.Random) -> dict[str, str]:
    # Build one replacement dictionary for template filling.
    return {
        "amount": random_generator.choice(amount_values),
        "bank_name": random_generator.choice(bank_names),
        "brand": random_generator.choice(prize_brands),
        "case_number": random_generator.choice(case_numbers),
        "delivery_company": random_generator.choice(delivery_companies),
        "government_source": random_generator.choice(government_sources),
        "link": random_generator.choice(example_links),
        "name": random_generator.choice(person_names),
        "points": random_generator.choice(point_values),
        "reference_code": random_generator.choice(reference_codes),
        "time_window": random_generator.choice(time_windows),
        "tracking_code": random_generator.choice(tracking_codes),
    }



# main generator


def generate_synthetic_scam_messages(
    total_message_count: int,
    random_seed: int,
) -> list[dict[str, object]]:
    # Generate synthetic scam messages for the dataset. Notes: All generated rows are labeled as scam, all generated rows are clearly marked as synthetic, and we try to avoid exact duplicates.
    
    random_generator = random.Random(random_seed)

    subtype_names = list(scam_templates.keys())
    synthetic_records: list[dict[str, object]] = []
    used_messages: set[str] = set()

    for message_index in range(total_message_count):
        subtype_name = subtype_names[message_index % len(subtype_names)]
        message_text = ""

        # Try multiple times so we do not keep repeating the same message.
        for _ in range(25):
            template = random_generator.choice(scam_templates[subtype_name])
            generation_context = build_generation_context(random_generator)
            candidate_message_text = template.format(**generation_context)

            if candidate_message_text not in used_messages:
                message_text = candidate_message_text
                used_messages.add(candidate_message_text)
                break

        if not message_text:
            continue

        synthetic_records.append(
            {
                "message_text": message_text,
                "label": 1,
                "label_name": "scam",
                "source_name": synthetic_source_name,
                "source_url": synthetic_source_url,
                "data_origin_type": "synthetic_generation",
                "is_synthetic": True,
                "split": "",
                "scam_subtype": subtype_name,
            }
        )

    return synthetic_records