from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


# ------------------------------------------------------------
# project paths
# ------------------------------------------------------------

project_root_directory = Path(__file__).resolve().parent

data_directory = project_root_directory / "data"
raw_data_directory = data_directory / "raw"
downloads_directory = raw_data_directory / "downloads"
processed_data_directory = data_directory / "processed"
metadata_directory = data_directory / "metadata"
logs_directory = project_root_directory / "logs"


# ------------------------------------------------------------
# output file names
# ------------------------------------------------------------

raw_dataset_file_name = "raw_collected_dataset.csv"
full_dataset_file_name = "full_dataset.csv"
train_dataset_file_name = "train_dataset.csv"
test_dataset_file_name = "test_dataset.csv"
metadata_file_name = "dataset_metadata.json"
source_validation_log_file_name = "source_validation_log.json"
skipped_rows_log_file_name = "skipped_rows_log.json"


# ------------------------------------------------------------
# general settings
# ------------------------------------------------------------

request_timeout_seconds = 20
random_seed = 42

train_split_ratio = 0.80
test_split_ratio = 0.20

minimum_message_length = 3
use_stopword_removal = False

user_agent = "sms_scam_detector_dataset_builder/1.0"


# ------------------------------------------------------------
# synthetic data settings
# ------------------------------------------------------------

synthetic_scam_message_count = 72
synthetic_source_name = "synthetic_rule_based_scam_messages"
synthetic_source_url = "synthetic_rule_based_generation"


# ------------------------------------------------------------
# dataset columns
# ------------------------------------------------------------

required_dataset_columns = [
    "message_text",
    "label",
    "label_name",
    "source_name",
    "source_url",
    "data_origin_type",
    "is_synthetic",
    "split",
    "cleaned_message_text",
    "annotation_count",
    "risk_categories_present",
    "suspicious_phrases_json",
]

final_dataset_column_order = [
    "message_text",
    "cleaned_message_text",
    "unigram_bigram_ready_text",
    "annotation_count",
    "risk_categories_present",
    "suspicious_phrases_json",
    "label",
    "label_name",
    "scam_subtype",
    "source_name",
    "source_url",
    "data_origin_type",
    "is_synthetic",
    "split",
    "message_length_characters",
    "message_length_words",
    "cleaned_token_count",
    "digit_count",
    "exclamation_mark_count",
    "contains_url_like_text",
    "contains_phone_number_like_text",
]


# ------------------------------------------------------------
# ML helper settings
# ------------------------------------------------------------

default_training_text_column = "unigram_bigram_ready_text"
default_label_column = "label"

default_vectorizer_configuration = {
    "vectorizer_type": "TfidfVectorizer",
    "analyzer": "word",
    "ngram_range": [1, 2],
    "lowercase": False,
    "min_df": 1,
    "token_pattern": r"(?u)\b\w+\b",
}


# ------------------------------------------------------------
# annotation settings
# ------------------------------------------------------------

risk_category_output_delimiter = "|"


# ------------------------------------------------------------
# regex patterns for suspicious phrases
# ------------------------------------------------------------

risk_category_regex_patterns = {
    "urgency": (
        r"\burgent\b",
        r"\bact now\b",
        r"\bimmediate action needed\b",
        r"\bfinal reminder\b",
        r"\bfinal notice\b",
        r"\brespond immediately\b",
        r"\bupdate immediately\b",
    ),
    "threat": (
        r"\blegal action\b",
        r"\bcourt notice\b",
        r"\benforcement action\b",
        r"\badditional penalties\b",
        r"\baccount will be locked\b",
        r"\bdriving privileges will be suspended\b",
        r"\blegal review\b",
        r"\bservice will be suspended\b",
    ),
    "payment_request": (
        r"\bpay now\b",
        r"\boverdue fee\b",
        r"\binvoice due\b",
        r"\bsubmit payment\b",
        r"\bbalance due\b",
        r"\bunpaid toll\b",
        r"\bprocessing fee\b",
        r"\bunpaid postage\b",
        r"\byou owe\b",
        r"\bservice charge\b",
        r"\bclear balance\b",
        r"\bpayment required\b",
    ),
    "account_verification": (
        r"\bverify your account\b",
        r"\bconfirm your identity\b",
        r"\bconfirm login\b",
        r"\bverify details\b",
        r"\bsecurity check\b",
        r"\bverify recent login\b",
        r"\bconfirm your details\b",
        r"\bsubmit verification\b",
        r"\bidentity verification\b",
    ),
    "impersonation": (
        r"\bUSPS\b",
        r"\bIRS\b",
        r"\bDMV\b",
        r"\bfraud team\b",
        r"\bsecurity team\b",
        r"\bthis is your manager\b",
        r"\bstate toll service\b",
        r"\btax office\b",
        r"\bcounty court\b",
    ),
    "prize_scam": (
        r"\byou won\b",
        r"\bclaim (?:your )?(?:reward|prize)\b",
        r"\bgift card\b",
        r"\breward points\b",
        r"\bcash bonus\b",
    ),
    "delivery_scam": (
        r"\bpackage on hold\b",
        r"\bdelivery failed\b",
        r"\btracking number\b",
        r"\bunpaid postage\b",
        r"\bshipment waiting\b",
        r"\bparcel notice\b",
    ),
    "link_request": (
        r"\bclick here\b",
        r"\bclick (?:the )?link\b",
        r"https?://\S+",
        r"www\.\S+",
    ),
    "contact_request": (
        r"\bcall now\b",
        r"\breply now\b",
        r"\bsend the code\b",
        r"\brespond now\b",
    ),
    "deadline_pressure": (
        r"\bbefore midnight\b",
        r"\bwithin 1 hour\b",
        r"\bwithin 24 hours\b",
        r"\bexpires today\b",
    ),
}


# ------------------------------------------------------------
# explanation templates
# ------------------------------------------------------------

risk_category_explanation_templates = {
    "urgency": "This phrase creates pressure to act quickly.",
    "threat": "This phrase uses fear or consequences.",
    "payment_request": "This phrase asks for money or payment.",
    "account_verification": "This phrase asks to verify account details.",
    "impersonation": "This suggests a trusted organization is being impersonated.",
    "prize_scam": "This promises a reward or prize.",
    "delivery_scam": "This refers to delivery issues commonly used in scams.",
    "link_request": "This pushes the user to click a link.",
    "contact_request": "This pushes the user to respond or send information.",
    "deadline_pressure": "This adds a time limit to pressure the user.",
}


# ------------------------------------------------------------
# extraction filters used by dataset_pipeline.py
# ------------------------------------------------------------

scam_example_cue_words = {
    "account",
    "amazon",
    "approved",
    "bank",
    "bonus",
    "claim",
    "click",
    "confirm",
    "court",
    "delivery",
    "dmv",
    "fee",
    "fine",
    "fraud",
    "gift",
    "inspection",
    "invoice",
    "irs",
    "link",
    "loan",
    "overdue",
    "package",
    "password",
    "pay",
    "payment",
    "points",
    "postage",
    "preapproved",
    "prize",
    "processed",
    "refund",
    "registration",
    "reward",
    "suspend",
    "ticket",
    "toll",
    "traffic",
    "urgent",
    "usps",
    "verify",
}

scam_action_words = {
    "act",
    "call",
    "claim",
    "click",
    "confirm",
    "log",
    "pay",
    "redeem",
    "reply",
    "respond",
    "scan",
    "send",
    "submit",
    "tap",
    "update",
    "verify",
}

message_filter_stop_prefixes = {
    "be cautious",
    "check it out first",
    "did you get",
    "don’t click",
    "don't click",
    "here are",
    "here’s how",
    "here's how",
    "if you clicked",
    "if you get",
    "if you think",
    "know that",
    "learn more",
    "never click",
    "only scammers",
    "report and delete",
    "search for",
    "stop and think",
    "take these steps",
    "to avoid",
    "turn on",
    "update your phone",
}

message_filter_stop_contains = {
    "accidentally do click",
    "consumer alert",
    "here are some other ways",
    "how these scams work",
    "how to avoid",
    "learn how to",
    "protect yourself",
    "report junk",
    "spot a scam",
    "the real irs",
    "want to know more",
}


# ------------------------------------------------------------
# source definition
# ------------------------------------------------------------

@dataclass(frozen=True)
class SourceDefinition:
    source_name: str
    source_type: str
    source_url: str
    enabled: bool
    permission_status: str
    permission_basis: str
    expected_license: str
    requires_robots_check: bool = False
    downloadable_file_web_address: str | None = None
    page_web_addresses: tuple[str, ...] = ()
    notes: str = ""


# ------------------------------------------------------------
# source manifest used by dataset_pipeline.py
# ------------------------------------------------------------

source_manifest: tuple[SourceDefinition, ...] = (
    SourceDefinition(
        source_name="uc_irvine_sms_spam_collection",
        source_type="downloadable_dataset",
        source_url="https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        downloadable_file_web_address="https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        enabled=True,
        permission_status="allowed",
        permission_basis="Official UCI Machine Learning Repository dataset page with a public CC BY 4.0 license.",
        expected_license="CC BY 4.0",
        notes="Main real-message dataset with safe and scam labels.",
    ),
    SourceDefinition(
        source_name="federal_trade_commission_consumer_text_scam_examples",
        source_type="web_page_examples",
        source_url="https://consumer.ftc.gov/unwanted-calls-emails-and-texts/unwanted-emails-texts-and-mail",
        enabled=True,
        permission_status="allowed",
        permission_basis="Official U.S. government public-interest pages. Only scrape when robots.txt allows it.",
        expected_license="Public U.S. government material subject to page access rules",
        requires_robots_check=True,
        page_web_addresses=(
            "https://consumer.ftc.gov/consumer-alerts/2020/09/heard-about-waiting-package-phishing-scam",
            "https://consumer.ftc.gov/consumer-alerts/2022/08/dont-click-random-text-its-scam",
            "https://consumer.ftc.gov/consumer-alerts/2024/01/irs-doesnt-send-tax-refunds-email-or-text",
            "https://consumer.ftc.gov/consumer-alerts/2024/05/text-about-overdue-toll-charges-probably-scam",
            "https://consumer.ftc.gov/consumer-alerts/2025/04/think-text-message-usps-it-could-be-scam",
        ),
        notes="Supplemental public scam-text examples.",
    ),
    SourceDefinition(
        source_name="competition_bureau_canada_text_scam_examples",
        source_type="web_page_examples",
        source_url="https://www.canada.ca/en/competition-bureau/news/2025/07/ding-scammers-are-hiding-in-your-text-messages.html",
        enabled=True,
        permission_status="allowed",
        permission_basis="Official Government of Canada public page. Only scrape when robots.txt allows it.",
        expected_license="Public government page subject to page access rules",
        requires_robots_check=True,
        page_web_addresses=(
            "https://www.canada.ca/en/competition-bureau/news/2025/07/ding-scammers-are-hiding-in-your-text-messages.html",
        ),
        notes="Supplemental public scam-text examples.",
    ),
)