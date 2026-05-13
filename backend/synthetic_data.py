from __future__ import annotations

import random

from config import (
    synthetic_safe_source_name,
    synthetic_safe_source_url,
    synthetic_source_name,
    synthetic_source_url,
)


#Simple word lists --------------------------------------
#These make the fake messages a little less copy-paste looking.

#Simple word lists --------------------------------------
#These make the fake messages a little less copy-paste looking.

brands = ["Target", "Costco", "Amazon", "Walmart"]
banks = ["RiverBank", "Pioneer Credit", "Union Trust", "North Valley Bank"]
delivery_companies = ["USPS", "UPS", "FedEx", "DHL"]
gov_places = ["DMV", "county court", "state toll service", "tax office"]
people = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley"]

amounts = ["$4.99", "$12.80", "$35.00", "$89.20", "$250.00", "$1,000.00"]
points = ["480", "900", "1,250", "2,400", "3,000"]
cases = ["C-1842", "T-5501", "A-9033", "P-7710", "R-4418", "V-6624"]
refs = ["REF284", "REF510", "REF773", "REF812", "REF945", "REF991"]
tracks = ["TRK3812", "TRK5921", "TRK6640", "TRK7425", "TRK8804", "TRK9317"]

times = ["before 2 PM", "within 1 hour", "today", "before midnight", "within 30 minutes"]

links = [
    "https://example.com/verify",
    "https://example.com/claim",
    "https://example.com/pay",
    "https://example.com/update",
]


#Scam templates --------------------------------------
#These are intentionally kind of obvious because they teach the model.

scam_templates = {
    "impersonation": [
        "{name}, this is your manager. Buy 3 gift cards {time} and send the codes here. Ref {ref}.",
        "Notice from {gov}: your case {case} is pending review. Confirm your identity at {link}.",
        "Your {bank} fraud team needs you to verify recent login {ref} now at {link}.",
        "This is your manager. Send the code now so I can approve payroll.",
    ],
    "urgency": [
        "Urgent: your account will be locked {time} unless you confirm your details at {link}.",
        "Final reminder: act now to avoid service interruption on case {case}. Update your billing profile at {link}.",
        "Immediate action needed. Your benefits expire {time} unless you submit verification now.",
        "Account will be locked.",
        "Verify now or your account may be suspended.",
    ],
    "threat": [
        "Final notice from {gov}: unpaid balance recorded under {case}. Failure to respond may trigger enforcement action.",
        "Your driving privileges will be suspended if you do not settle violation {case} immediately at {link}.",
        "Court alert: missing payment for case {case} may result in additional penalties and legal review.",
        "Legal action pending unless payment is received today.",
        "Service will be suspended unless you clear balance now.",
    ],
    "payment_request": [
        "You owe an overdue fee of {amount}. Pay now to avoid added charges on {ref}: {link}.",
        "Invoice due today for your account balance of {amount}. Submit payment for {ref} at {link}.",
        "A service charge of {amount} is pending. Clear balance {ref} now to prevent a hold.",
        "Payment required today.",
        "Pay now to remove the overdue fee.",
    ],
    "prize_scam": [
        "Congratulations! You won a {brand} gift card. Claim prize {ref} now at {link}.",
        "You were selected for a cash bonus of {amount}. Confirm reward {ref} {time}.",
        "Your loyalty account has {points} expiring points. Redeem batch {ref} now at {link}.",
        "You won a gift card. Claim your reward now.",
    ],
    "account_verification": [
        "Security alert: verify your {bank} profile now to stop unauthorized access on {ref}.",
        "We detected unusual activity on your account. Confirm login {ref} immediately at {link}.",
        "Your payment app needs quick account verification for transfer batch {ref} before service can continue.",
        "Verify your account now.",
        "Confirm your identity to keep your account open.",
    ],
    "delivery_scam": [
        "{delivery}: package {track} is on hold due to unpaid postage of {amount}. Pay at {link}.",
        "{delivery} delivery failed for shipment {track}. Update shipping preferences now using {link}.",
        "Parcel notice: shipment {track} is waiting for address confirmation before final delivery.",
        "Your package is on hold. Pay unpaid postage now.",
    ],
}


#Safe templates --------------------------------------
#These help the model chill out a bit and not call every normal text a scam.

safe_templates = [
    "Hey, are we still meeting after class?",
    "I am running late but I am on the way.",
    "Can you send me the notes from today?",
    "Your order was delivered. Thanks for shopping with us.",
    "Your bank statement is ready in the app.",
    "Your account settings were updated successfully.",
    "Reminder: your appointment is tomorrow at 2 PM.",
    "The package was left at your front door.",
    "Your verification code is {ref}. Do not share it with anyone.",
    "Your payment was received. No action is needed.",
    "Your password was changed successfully.",
    "Your refund has been processed.",
    "Class starts at 10 tomorrow, don't be late lol.",
    "Can you pick up coffee before you come over?",
    "Your delivery is scheduled for today.",
    "Your account balance is available in the mobile app.",
]

#Helper stuff --------------------------------------

def build_generation_context(random_generator: random.Random) -> dict[str, str]:
    #Make one small dictionary for filling in templates.
    return {
        "amount": rng.choice(amounts),
        "bank": rng.choice(banks),
        "brand": rng.choice(brands),
        "case": rng.choice(cases),
        "delivery": rng.choice(delivery_companies),
        "gov": rng.choice(gov_places),
        "link": rng.choice(links),
        "name": rng.choice(people),
        "points": rng.choice(points),
        "ref": rng.choice(refs),
        "time": rng.choice(times),
        "track": rng.choice(tracks),
    }


def build_synthetic_record(message_text: str, label: int, source_name: str, source_url: str, subtype: str) -> dict[str, object]:
    #Put one synthetic row into the same shape as the real data.
    return {
        "message_text": message_text,
        "label": int(label),
        "label_name": "scam" if int(label) == 1 else "safe",
        "source_name": source_name,
        "source_url": source_url,
        "data_origin_type": "synthetic_generation",
        "is_synthetic": True,
        "split": "",
        "scam_subtype": subtype,
    }
    
#Main generators --------------------------------------

def generate_synthetic_scam_messages(
    total_message_count: int,
    random_seed: int,
) -> list[dict[str, object]]:
    #Make scam rows for training.
    rng = random.Random(random_seed)
    subtype_names = list(scam_templates.keys())
    
    rows: list[dict[str, object]] = []
    used_messages: set[str] = set()

    for message_index in range(total_message_count):
        subtype = subtype_names[message_index % len(subtype_names)]
        message_text = ""

        # Try multiple times so we do not keep repeating the same message.
        for _ in range(40):
            template = rng.choice(scam_templates[subtype])
            context = build_generation_context(rng)
            maybe_message = template.format(**context)

            if maybe_message not in used_messages:
                message_text = maybe_message
                used_messages.add(maybe_message)
                break

        if message_text:
            rows.append(
                build_synthetic_record(
                    message_text=message_text,
                    label=1,
                    source_name=synthetic_source_name,
                    source_url=synthetic_source_url,
                    subtype=subtype,
                )
            )

    return rows


def generate_synthetic_safe_messages(total_message_count: int, random_seed: int) -> list[dict[str, object]]:
    #Make normal rows too, because false positives are super annoying.
    rng = random.Random(random_seed + 1000)

    rows: list[dict[str, object]] = []
    used_messages: set[str] = set()

    for _ in range(total_message_count):
        message_text = ""

        for _ in range(40):
            template = rng.choice(safe_templates)
            context = build_generation_context(rng)
            maybe_message = template.format(**context)

            if maybe_message not in used_messages:
                message_text = maybe_message
                used_messages.add(message_text)
                break

        if message_text:
            rows.append(
                build_synthetic_record(
                    message_text=message_text,
                    label=0,
                    source_name=synthetic_safe_source_name,
                    source_url=synthetic_safe_source_url,
                    subtype="none",
                )
            )

    return rows
