# Proposal: weak

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
| Intended level | `weak` |
| Expected `overall_fit` | 5-22 (inclusive) |
| Expected high | _(none)_ |
| Expected low | `skills_match`, `experience_match`, `role_relevance`, `overall_fit` |

**Rationale (the generator's own words):**

Marguerite is a coherent, experienced front-end engineer — a genuinely employable professional — but she is a poor fit for a senior backend payments role on essentially every axis. Her primary languages are JavaScript/TypeScript, not Python or Go. Her database exposure is explicitly described as ad-hoc queries with no schema design or production ownership. She has no backend service design, no distributed-systems experience, no payments-rail or ledger work, and no on-call/incident ownership. The one payments-adjacent touch (Stripe Checkout embeds at Nordvale) is explicitly framed as configuring hosted widgets, not integration work. Her six years of tenure rules out a simple "not enough experience" dismissal, making the mismatch clearly one of domain and technology stack rather than seniority, which is exactly the weak-fit profile requested.

## Resume

<!-- Verbatim resume_markdown. This is the text the scorer would
     read; do not reformat it when reviewing. -->

# Marguerite Solano
marguerite.solano@emailbox.net · (503) 774-2091 · Portland, OR · github.com/msolano-dev

---

## Summary

Front-end focused software engineer with 6 years of experience building consumer-facing web applications and design systems. Passionate about crafting intuitive user interfaces, accessibility, and cross-browser compatibility. Skilled in React, TypeScript, and CSS tooling, with a track record of shipping polished features in fast-moving product teams.

---

## Experience

**Senior Frontend Engineer — Luminary Creative Agency**
*Portland, OR · March 2021 – Present*

- Led the redesign of an e-commerce storefront for a mid-sized fashion retailer, resulting in a 22% improvement in conversion rate.
- Built and maintained a shared component library (React + Storybook) used across five product squads.
- Collaborated with backend engineers to integrate REST APIs for product catalog, cart, and checkout flows — primarily a consumer of APIs, not a designer.
- Introduced automated accessibility auditing (axe-core) into the CI pipeline, bringing WCAG 2.1 AA compliance to all new page templates.
- Mentored two junior front-end engineers on component architecture and code review practices.

**Frontend Engineer — Cartwheel Interactive**
*Seattle, WA · June 2018 – February 2021*

- Developed interactive dashboards for a SaaS analytics product using React and D3.js, visualizing time-series data for marketing teams.
- Partnered with UX designers to translate Figma prototypes into production-ready React components.
- Wrote Jest and Cypress test suites covering critical user flows; improved coverage from 34% to 71%.
- Helped migrate legacy jQuery codebase to React over 18 months, coordinating with four other engineers.
- Participated in bi-weekly sprint ceremonies and maintained project documentation in Confluence.

**Junior Web Developer — Nordvale Digital**
*Seattle, WA · August 2016 – May 2018*

- Built and maintained marketing websites for small-business clients using HTML, CSS, JavaScript, and WordPress.
- Implemented basic contact forms and payment widget embeds (Stripe Checkout) for client storefronts — configuring hosted widgets, not building custom integrations.
- Optimized page load performance via image compression and lazy loading, reducing average LCP by 1.4 seconds.

---

## Skills

- **Languages:** JavaScript (ES2020+), TypeScript, HTML, CSS/SCSS, some Python (scripting only)
- **Frameworks & Libraries:** React, Next.js, Storybook, D3.js, Tailwind CSS
- **Testing:** Jest, React Testing Library, Cypress
- **Tooling:** Webpack, Vite, ESLint, Prettier, GitHub Actions
- **Design collaboration:** Figma, Zeplin
- **Databases:** Basic SQL queries (SELECT, JOIN) for ad-hoc data pulls; no schema design or production DBA experience
- **Cloud:** Familiar with Vercel and Netlify deployments; limited AWS exposure (S3, CloudFront)

---

## Education

**B.S. in Digital Media Design**
Pacific Northwest College of Art & Technology, Portland, OR — 2016

---

## Side Projects

- **a11y-linter:** A small open-source CLI tool that runs accessibility checks against static HTML files. ~180 GitHub stars.
- **Budget Buddy:** A personal finance tracker built with Next.js and a local SQLite database, intended for household expense categorization.
