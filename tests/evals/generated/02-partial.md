# Proposal: partial

> **Unreviewed, machine-generated proposal — not an eval case.**
>
> An LLM wrote this resume and then stated what it was aiming for. The
> level, score range and dimension expectations below are that stated
> intent: they have not been checked against the scorer and no human has
> agreed with them. Nothing here runs in the eval suite. Review this
> proposal before any part of it is trusted.

## Stated generation intent

| Field | Proposed value |
| --- | --- |
| Intended level | `partial` |
| Expected `overall_fit` | 42-62 (inclusive) |
| Expected high | `skills_match`, `experience_match` |
| Expected low | `role_relevance`, `overall_fit` |

**Rationale (the generator's own words):**

Priya has solid Python, strong PostgreSQL depth, real REST API ownership, and genuine on-call/postmortem experience — so experience_match and skills_match should land reasonably well. However, her payments exposure is shallow (Stripe webhooks at the application layer, no direct card-network, ACH, or acquirer-level work), she has no Go experience, distributed-systems fundamentals are only lightly evidenced (idempotency keys mentioned once, no circuit breakers, no event streaming beyond SQS), and her billing background is e-commerce subscriptions rather than a true payments-platform context. A hiring team would credibly argue both sides: she's not a clear reject but there are real gaps in the core domain and in infrastructure maturity.

## Resume

<!-- Verbatim resume_markdown. This is the text the scorer would
     read; do not reformat it when reviewing. -->

# Priya Nambiar
priya.nambiar@email.com | linkedin.com/in/priyanambiar | Austin, TX

## Summary
Backend engineer with 6 years of experience building data-intensive services in Python and Java. Most of my career has been in e-commerce and SaaS, with recent exposure to billing and subscription systems. I enjoy thinking carefully about API design and data modeling, and I've gotten increasingly comfortable with production operations over the last couple of years.

---

## Experience

### Senior Software Engineer — Cartwell Commerce
*March 2021 – Present | Austin, TX*

Cartwell is a mid-market e-commerce platform serving ~800 online retailers.

- Led backend development of the subscription billing module, including proration logic, dunning workflows, and invoice generation. The module handles ~$4M in monthly recurring charges.
- Designed PostgreSQL schemas for billing events and invoices; introduced composite indexes and query rewrites that reduced p99 checkout latency by 35%.
- Built and maintained a REST API (Python/FastAPI) consumed by the frontend team and two third-party integrations; wrote the API contract, versioned endpoints, and maintained backwards compatibility through three major releases.
- Integrated with Stripe's payment API for card charging and refunds; handled webhook delivery, idempotency keys, and retry logic when Stripe returned transient errors.
- Participated in on-call rotation (one week in five); wrote postmortems for two significant incidents, one involving a double-charge bug traced to a missing unique constraint, and one involving webhook processing delays under high load.
- Onboarded and mentored two junior engineers; ran weekly code review sessions.

### Software Engineer — Nodebridge Analytics
*July 2018 – February 2021 | Remote*

Nodebridge provided a SaaS analytics platform for retail chains.

- Built ETL pipelines in Python (Celery + PostgreSQL) to ingest and normalize point-of-sale data from ~200 retail locations.
- Contributed to the REST API layer used by customer dashboards; implemented pagination, filtering, and rate limiting.
- Improved database performance through index analysis and vacuuming strategies on a multi-tenant PostgreSQL instance (~800 GB).
- Added basic retry and dead-letter-queue handling to Celery workers to improve reliability of background jobs.
- No on-call responsibilities in this role; production issues were handled by a separate SRE team.

---

## Skills

- **Languages:** Python (primary), Java (working knowledge), SQL
- **Frameworks:** FastAPI, Django REST Framework, Flask, Celery
- **Databases:** PostgreSQL, Redis
- **Infrastructure:** Docker, basic AWS (EC2, RDS, SQS); limited Kubernetes experience
- **Messaging:** SQS, some familiarity with Kafka (personal projects)
- **Payments:** Stripe API, basic webhook handling
- **Other:** REST API design, OpenAPI/Swagger, pytest, GitHub Actions

---

## Education

**B.S. Computer Science** — University of Texas at San Antonio
*Graduated May 2018*

---

## Notable Projects

- **Billing retry engine (Cartwell):** Designed a state-machine-based retry system for failed charges, with configurable backoff schedules and merchant-level settings. Reduced involuntary churn by ~12%.
- **Invoice reconciliation tooling (Cartwell):** Built an internal CLI and reporting dashboard that compares Stripe payout reports against internal ledger records; surfaced ~$18k in previously undetected discrepancies over six months.
