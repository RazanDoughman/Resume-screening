# Proposal: adversarial

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
| Intended level | `adversarial` |
| Expected `overall_fit` | 38-62 (inclusive) |
| Expected high | `skills_match` |
| Expected low | `experience_match`, `role_relevance`, `overall_fit` |

**Rationale (the generator's own words):**

This resume is engineered to mirror the JD's vocabulary almost perfectly — "idempotency," "at-least-once delivery," "circuit breakers," "auth, capture, refund, reconciliation," "risk and integrations teams," "thousands of merchants," "postmortems," "on-call" — while hiding critical weaknesses underneath. The Payvelo role at the senior level makes sweeping, unfalsifiable claims ("architected and owned the core payments platform," "industry-leading reliability") with no concrete metrics, team size, or architectural decisions described. The actual transactional-systems depth at Cloudspark turns out to be analytics reporting and webhook ingestion — adjacent but not qualifying. Go is listed as merely "familiar" rather than production-ready. PCI DSS and SOC 2 appear as "awareness" and "exposure" only. A careful scorer should recognize that the skills section reads like a transcription of the JD's bullet points, and that the Payvelo accomplishments are inflated assertions with no substance to verify.

## Resume

<!-- Verbatim resume_markdown. This is the text the scorer would
     read; do not reformat it when reviewing. -->

# Dmitri Valenkov
dmitri.valenkov@emailhub.io · (415) 882-0341 · San Francisco, CA · linkedin.com/in/dmitrivalenkov

---

## Summary

Senior backend engineer with 6+ years building scalable, transactional systems for high-throughput payment platforms. Deep experience designing REST and gRPC APIs, owning production services end-to-end, and driving reliability improvements across distributed architectures. Passionate about idempotency, at-least-once delivery, and the full lifecycle of payment flows from auth to reconciliation.

---

## Experience

### Senior Backend Engineer — Payvelo Technologies
*San Francisco, CA · March 2021 – Present*

Payvelo is a payments infrastructure startup serving mid-market merchants with auth, capture, and refund workflows.

- Architected and owned the core payments platform, delivering industry-leading reliability for thousands of merchants processing millions of transactions.
- Led design of the company's REST and gRPC API layer, establishing patterns for idempotency, retries, and circuit breakers used across all backend services.
- Championed observability improvements across the platform by introducing structured logging and dashboards, dramatically reducing mean time to resolution for production incidents.
- Wrote postmortems and ran incident response for the payments team; established on-call rotation practices adopted company-wide.
- Mentored three mid-level engineers on distributed-systems fundamentals and PostgreSQL schema design best practices.
- Drove reconciliation and ledger service design, working closely with risk and integrations teams.

*Technologies: Python, PostgreSQL, Kafka, Kubernetes, Terraform, Stripe, gRPC*

---

### Backend Engineer — Cloudspark Solutions
*Austin, TX · June 2018 – February 2021*

Cloudspark builds SaaS tooling for e-commerce analytics and reporting.

- Built data pipeline services in Python that aggregated transaction records from multiple merchant sources into a centralized reporting database.
- Maintained PostgreSQL schemas for analytics workloads; added indexes and ran EXPLAIN plans to improve query performance on large datasets.
- Contributed to internal REST APIs consumed by the frontend team.
- Participated in bi-weekly on-call rotations for the data ingestion infrastructure.
- Supported integration with Stripe webhooks to ingest payment event data for reporting purposes.

*Technologies: Python, PostgreSQL, REST APIs, Stripe Webhooks, AWS, Docker*

---

### Junior Software Engineer — Brightloop Agency
*Austin, TX · August 2016 – May 2018*

Brightloop is a digital agency delivering custom web applications for small-business clients.

- Developed backend features for client-facing web applications using Django and Flask.
- Wrote SQL queries and maintained relational schemas for client project databases.
- Integrated third-party payment buttons (PayPal, Stripe Checkout) into client storefronts.
- Collaborated with front-end developers and project managers on 10+ client deliverables.

*Technologies: Python, Django, Flask, MySQL, JavaScript, PayPal API, Stripe Checkout*

---

## Skills

**Languages:** Python (primary), Go (familiar), SQL  
**Databases:** PostgreSQL, MySQL  
**Distributed Systems:** Idempotency, retries, at-least-once delivery, eventual consistency, circuit breakers  
**Messaging:** Kafka, AWS Kinesis  
**Infrastructure:** Kubernetes, Terraform, AWS, Docker  
**APIs:** REST, gRPC  
**Payments:** Stripe, ACH, card network concepts, PCI DSS awareness  
**Compliance:** PCI DSS, SOC 2 exposure  

---

## Education

**B.S. Computer Science** — University of North Texas, Denton, TX · May 2016

---

## Additional

- Speaker, internal engineering all-hands: "Designing for Idempotency in Payment Systems" (2022)
- Open-source contributor to a Python gRPC utilities library (personal GitHub)
