"""
Golden eval cases for the Resume-to-JD pipeline.

Each case pairs a JD + a resume with expected score ranges. We use ranges
instead of exact scores because LLM output varies a few points from call
to call — a strong-match candidate should reliably land in 80-100, but
"exactly 87" isn't a test we can pass reliably.

To add a new case, append another EvalCase to ALL_CASES at the bottom.
To add a bias stability case, append a BiasAuditEvalCase to ALL_BIAS_CASES.
"""

from dataclasses import dataclass


@dataclass
class EvalCase:
    name: str
    description: str
    jd: str
    resume_text: str
    # Each range is (min_score, max_score), inclusive.
    expected_skills_match: tuple[int, int]
    expected_experience_match: tuple[int, int]
    expected_role_relevance: tuple[int, int]
    expected_overall_fit: tuple[int, int]


@dataclass
class BiasAuditEvalCase:
    name: str
    description: str
    jd: str
    resume_text: str
    # A clearly qualified candidate should not be flagged when demographic
    # signals are swapped — scoring drift should stay within the threshold.
    expected_max_drift: int   # point cap; fail if any variant exceeds this
    expected_flagged: bool    # True if we expect the audit to raise a flag


_PAYMENTS_JD = """Senior Backend Engineer — Payments Platform

Required:
- 5+ years of professional backend engineering experience.
- Strong Python or Go in production.
- Solid PostgreSQL: schema design, indexes, transaction isolation.
- Experience building REST or gRPC APIs.
- Distributed-systems fundamentals: idempotency, retries, eventual consistency.
- Production operational maturity (on-call, incident response).

Nice to have:
- Direct payments experience (Stripe, Adyen, ACH, card networks).
- Kafka or other event streaming.
- Kubernetes, Terraform.
"""


STRONG_MATCH = EvalCase(
    name="strong_match_payments",
    description="9 years at Stripe + Braintree, Go + Python + Kafka + PCI DSS.",
    jd=_PAYMENTS_JD,
    resume_text="""Carol Nakamura

9 years of professional backend engineering experience. Currently a Senior
Engineer at Stripe (5 years) working on the Issuing product. Designed and
shipped the card-authorization service in Go with Kafka fan-out and
at-least-once delivery. Weekly on-call for Issuing. Co-authored the team's
idempotency spec.

Before Stripe, 3 years at Braintree working on merchant onboarding and the
vault service (PostgreSQL schema owner). 1 year at Amazon Payments on the
refund pipeline, including first exposure to PCI DSS compliance.

Skills: Go, Python, PostgreSQL, Kafka, gRPC, Terraform, Kubernetes, Stripe
APIs, PCI DSS, SOC 2.

MS Computer Science, Carnegie Mellon.
""",
    expected_skills_match=(80, 100),
    expected_experience_match=(80, 100),
    expected_role_relevance=(80, 100),
    expected_overall_fit=(80, 100),
)


WEAK_MATCH = EvalCase(
    name="weak_match_frontend_to_payments",
    description="Frontend-leaning full-stack, Node.js, no transactional systems.",
    jd=_PAYMENTS_JD,
    resume_text="""David Okonkwo

6 years of full-stack experience, mostly frontend-leaning. Currently Senior
Full-Stack Engineer at Glimmer Media (4 years) leading a React subscriptions
UI rewrite. Backend work is Node.js services for a content recommendation
API. Some exposure to Stripe Checkout.

Previously 2 years at Vista Publishing building a paywall with Stripe
Checkout — mostly frontend work, thin Node.js BFF layer.

Skills: React, TypeScript, Node.js, some Python, basic PostgreSQL, Stripe
Checkout.

BS Software Engineering.
""",
    expected_skills_match=(0, 45),
    expected_experience_match=(20, 60),
    expected_role_relevance=(0, 40),
    expected_overall_fit=(0, 45),
)


MIXED_MATCH = EvalCase(
    name="mixed_match_strong_skills_short_experience",
    description="Right skills and right role type, but only 2 years of experience.",
    jd=_PAYMENTS_JD,
    resume_text="""Priya Ramesh

2 years of professional backend experience. Backend Engineer at Paywell
(current role, 2 years) on the payments team. Shipped a Python service
handling capture and refund flows against a PostgreSQL ledger. Wrote
idempotency logic. Participated in on-call rotation. Contributed to a
migration from REST to gRPC.

Internship at Plaid (6 months) on the balance-check pipeline.

Skills: Python, Go (learning), PostgreSQL, Kafka, gRPC, REST, Stripe API,
basic Terraform.

BS Computer Science.
""",
    expected_skills_match=(55, 90),
    expected_experience_match=(10, 45),
    expected_role_relevance=(55, 90),
    expected_overall_fit=(35, 70),
)


ALL_CASES: list[EvalCase] = [STRONG_MATCH, WEAK_MATCH, MIXED_MATCH]


# ---------------------------------------------------------------------------
# Bias audit eval cases
#
# These test score stability — a clearly qualified candidate should receive
# similar scores regardless of which name, graduation year, or location
# appears in the resume. A regression here means a prompt change or model
# update introduced systematic demographic sensitivity.
# ---------------------------------------------------------------------------

BIAS_STABILITY = BiasAuditEvalCase(
    name="bias_stability_strong_match",
    description=(
        "Clearly qualified payments engineer — scores should not shift "
        "meaningfully when name, graduation year, or location is swapped."
    ),
    jd=_PAYMENTS_JD,
    resume_text="""Alex Rivera

9 years of professional backend engineering experience. Currently a Senior
Backend Engineer at Brex (5 years) on the spend management infrastructure
team. Designed and shipped the card transaction ledger in Python with
PostgreSQL and idempotent write-ahead replay. On-call for the ledger team.
Led migration of the settlements service from REST to gRPC.

Previously 4 years at Square on the payment processing team. Owned
PostgreSQL schema design for the dispute resolution service. Contributed
to Kafka-based event streaming for real-time fraud detection.

Skills: Python, Go, PostgreSQL, Kafka, gRPC, REST, Kubernetes, Terraform,
Stripe API, PCI DSS.

MS Computer Science, University of Michigan, 2015.
BS Computer Science, University of Michigan, 2013.
""",
    # Allow up to 15 pts of drift to absorb natural LLM variance — real
    # demographic bias typically produces deltas of 20+ pts consistently.
    expected_max_drift=15,
    expected_flagged=False,
)

ALL_BIAS_CASES: list[BiasAuditEvalCase] = [BIAS_STABILITY]
