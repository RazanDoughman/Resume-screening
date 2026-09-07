# Proposal: strong

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
| Intended level | `strong` |
| Expected `overall_fit` | 78-93 (inclusive) |
| Expected high | `skills_match`, `experience_match`, `role_relevance`, `overall_fit` |
| Expected low | _(none)_ |

**Rationale (the generator's own words):**

This candidate has 7+ years of backend experience, the majority spent specifically on payment flows (auth, refund, reconciliation, ACH) — directly matching the JD's core domain. She demonstrates deep Python production use, PostgreSQL expertise with concrete specifics (transaction isolation, index tuning, serializable mode), and gRPC/REST API design for cross-team consumers. She satisfies most nice-to-haves as well: direct Stripe and Adyen experience, Kafka, Kubernetes/Terraform, PCI DSS and SOC 2 exposure, and mentorship of mid-level engineers. The one realistic imperfection is that Go is listed as only "working knowledge," not a primary language — giving a scorer something to mark down without undermining the overall strong fit. Every claimed skill is grounded in a specific employer context, concrete metric, or named technology, so the substance holds up to scrutiny.

## Resume

<!-- Verbatim resume_markdown. This is the text the scorer would
     read; do not reformat it when reviewing. -->

# Priya Subramaniam
priya.subramaniam@email.com | (415) 882-0134 | linkedin.com/in/priyasub | San Francisco, CA

---

## Summary

Backend engineer with 7 years of experience building reliable, high-throughput transactional systems at fintech and payments companies. Deep hands-on background in Python and PostgreSQL, with a strong track record designing payment flows—auth, capture, refund, settlement—at scale. Comfortable owning services end-to-end: from schema design through API contract to on-call. Known for clear written communication and raising the quality bar through design review and mentorship.

---

## Experience

### Senior Software Engineer — Payments Platform
**Meridian Payments Inc.** | San Francisco, CA | June 2020 – Present

Meridian provides a payment orchestration layer used by ~4,000 SMB and enterprise merchants across North America.

- **Led end-to-end design and implementation of the refund and dispute service** (Python/FastAPI, PostgreSQL 14), handling ~2.5 M refund events/month. Defined the database schema, idempotency key strategy, and gRPC API surface consumed by three downstream teams.
- Reduced median refund settlement latency from 8 minutes to 90 seconds by rearchitecting the retry/backoff logic and adding Kafka-based event streaming for async dispute status updates, eliminating polling against the card network adapter.
- Designed and rolled out multi-currency ledger reconciliation pipeline using PostgreSQL advisory locks and serializable transaction isolation, eliminating a class of double-credit bugs that had been in production for two years.
- Authored and triaged postmortems for six production incidents (P0/P1); implemented circuit breakers (using `tenacity` and a lightweight Redis-backed state store) on the Adyen and Stripe adapter layer, cutting cascading failure events by ~70% over 12 months.
- Participated in on-call rotation (1 week in 5); drove SLO definition for the refund service, establishing 99.95% availability and p99 latency targets tracked in Datadog.
- Mentored two mid-level engineers through architecture design reviews and weekly 1:1s; one was promoted to senior within 14 months.
- Worked with compliance team during company's PCI DSS Level 1 audit; scoped and implemented data-at-rest encryption for cardholder data fields and maintained audit log table with immutable append semantics.

**Key technologies:** Python 3.10, FastAPI, gRPC, PostgreSQL 14, Kafka, Redis, Kubernetes (GKE), Terraform, Datadog, Adyen API, Stripe API

---

### Software Engineer — Core Infrastructure
**Cobalt Financial Systems** | New York, NY | August 2017 – May 2020

Cobalt builds white-label banking and bill-payment infrastructure for credit unions.

- Built and maintained a Python (Django + Celery) ACH origination service processing ~$200 M in monthly transaction volume. Implemented NACHA file generation, R-code handling, and same-day ACH cutoff scheduling.
- Designed the initial schema and query optimization strategy for a PostgreSQL-backed general ledger (double-entry, multi-currency), including composite indexes on `(account_id, posted_at)` and partial indexes for unsettled entries; reduced P95 balance-query time from 340 ms to 22 ms.
- Contributed to migration of monolith ACH module to a standalone service, writing the REST API used by two partner credit unions' core banking systems.
- Participated in SOC 2 Type II readiness efforts: helped scope audit log requirements and implemented structured JSON logging with field-level redaction.
- On-call for ACH service; wrote 4 postmortems over 2.5 years, each with root cause analysis and preventive follow-up items tracked to closure.

**Key technologies:** Python 3.7, Django, Celery, PostgreSQL 12, RabbitMQ, AWS (ECS, RDS, SQS), Terraform, Splunk

---

### Junior Software Engineer
**Greenline Labs** | Austin, TX | July 2016 – July 2017

Early-stage SaaS startup building subscription billing software for mid-market retailers.

- Implemented billing cycle engine (Python) for recurring charge scheduling with proration logic and failed-payment retry policies.
- Built internal REST API (Flask) for subscription CRUD operations; wrote integration tests against a live Stripe test environment.
- Maintained PostgreSQL schema for subscriber, invoice, and payment_method tables.

---

## Skills

- **Languages:** Python (expert), Go (working knowledge), SQL
- **Databases:** PostgreSQL (schema design, MVCC, index tuning, replication), Redis
- **APIs / Protocols:** REST, gRPC, OpenAPI/Swagger
- **Streaming:** Kafka, AWS SQS/SNS
- **Infrastructure:** Kubernetes (GKE, EKS), Terraform, Docker, GitHub Actions
- **Observability:** Datadog, Prometheus/Grafana, structured logging (OpenTelemetry)
- **Payment Rails:** Stripe, Adyen, ACH/NACHA
- **Compliance:** PCI DSS (Level 1 participation), SOC 2 Type II readiness

---

## Education

**B.S. Computer Science** — University of Texas at Austin | May 2016
Relevant coursework: Databases, Distributed Systems, Operating Systems

---

## Selected Writing & Talks

- Internal tech talk: "Idempotency Keys in Practice — Lessons from 50 Million Refund Requests" (Meridian Eng All-Hands, 2022)
- Engineering blog post: "Double-entry ledgers in PostgreSQL without losing your mind" (Meridian Engineering Blog, 2023)
