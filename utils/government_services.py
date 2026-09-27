"""
utils/government_services.py
-----------------------------
Renders the "Government Services" page: a static, informational directory
of official Government of India student/career services (internships,
skilling, scholarships, digital documents, academic identity, apprenticeships,
career services, and education loans).

CONTENT SOURCE: every service name, description, URL, and note below is
carried over from the supplied reference file (government_services_page.py)
without invention or alteration of any government-related fact. Two purely
self-referential edits were made for accuracy in THIS app (see notes at each
site below); no government scheme, eligibility rule, deadline, or URL was
changed, added, or removed.

Deliberately static (no scraping, no external API calls, no new
dependencies, no data collection) - the services below are the entire
content model. Rendering was adapted to LearnMate Analytics AI's own
existing card/badge CSS classes (.lm-card / .lm-badge, defined in app.py)
instead of the reference file's `frontend.components` helpers
(hero() / glass_card_open() / glass_card_close()), which are not part of
this project.

LearnMate Analytics AI is NOT an official Government of India representative;
this page only links out to official portals and never collects, submits,
or stores any government-related data.
"""

from __future__ import annotations

import streamlit as st

# Each entry: icon, title, what it is, who it's for, how to access it, url, [note]
# Carried over verbatim from the supplied source file. `url` is only omitted
# (falls back to a "pending verification" caption instead of a link) if a
# link was not confidently supplied in the source - no URL is ever guessed.
GOVERNMENT_SERVICES: list[dict] = [
    {
        "icon": "\U0001F1EE\U0001F1F3",
        "title": "AICTE National Internship Portal",
        "what": "A platform for students and fresh engineers to discover internship opportunities posted through the AICTE ecosystem.",
        "who": "Students and eligible learners looking for internship opportunities, subject to the requirements of individual listings.",
        "how": "Create or complete a student profile, explore available internships, review eligibility and deadlines, and apply through the official portal.",
        "url": "https://internship.aicte-india.org/",
    },
    {
        "icon": "\U0001F393",
        "title": "Skill India Digital Hub (SIDH)",
        "what": "A Government of India digital platform for discovering skill-development, training and related career opportunities.",
        "who": "Students, learners, job seekers and individuals looking to build or improve industry-relevant skills.",
        "how": "Visit the official SIDH portal, create/login to your learner profile, explore available opportunities and check the eligibility and current terms for each program.",
        "url": "https://www.skillindiadigital.gov.in/",
    },
    {
        "icon": "\U0001F4B0",
        "title": "National Scholarship Portal (NSP)",
        "what": "A Government of India platform where students can discover and apply for scholarship schemes available through participating government departments and organizations.",
        "who": "Students who meet the eligibility requirements of individual scholarship schemes.",
        "how": "Create the required student registration/OTR credentials where applicable, review available schemes, check eligibility, and submit applications through the official portal.",
        "url": "https://scholarships.gov.in/",
        "note": "Scheme eligibility, deadlines and academic-year details are not hard-coded here since they change every cycle - always check the current cycle on the official portal.",
    },
    {
        "icon": "\U0001F4C4",
        "title": "DigiLocker",
        "what": "A Government of India digital document platform (a flagship MeitY initiative under Digital India) that provides access to authentic digital documents and certificates.",
        "who": "Students and citizens who need to access, store, share or verify eligible digital documents.",
        "how": "Create/login to DigiLocker, verify your identity as required, and fetch eligible documents from participating issuers.",
        "url": "https://www.digilocker.gov.in/",
        "note": "Not every document is automatically available - availability depends on the issuing authority.",
    },
    {
        "icon": "\U0001F194",
        "title": "APAAR / Academic Bank of Credits (ABC)",
        "what": "APAAR (Automated Permanent Academic Account Registry) provides students with a persistent academic identity and supports the management and recognition of academic achievements and credits. It integrates with DigiLocker and the Academic Bank of Credits ecosystem.",
        "who": "Students participating in India's academic ecosystem, subject to the applicable processes of their institution and the official system.",
        "how": "Follow the official APAAR process and institution-supported steps for creation and access.",
        "url": "https://apaar.education.gov.in/",
        # Self-referential edit: "LearnMate AI" -> "LearnMate Analytics AI"
        # to match this app's actual product name. No factual claim changed.
        "note": "LearnMate Analytics AI cannot create or modify a student's APAAR ID - this is guidance only.",
    },
    {
        "icon": "\U0001F3E2",
        "title": "National Apprenticeship Training Scheme (NATS)",
        "what": "Provides practical, on-the-job training opportunities for eligible students and graduates through apprenticeship programs.",
        "who": "Eligible graduate, diploma and vocational certificate holders, depending on the applicable NATS rules and opportunities.",
        "how": "Register on the official NATS portal, complete your profile, explore apprenticeship opportunities, review eligibility requirements and apply to suitable opportunities.",
        "url": "https://nats.education.gov.in/",
        "note": "Completing an apprenticeship does not guarantee employment.",
    },
    {
        "icon": "\U0001F4BC",
        "title": "National Career Service (NCS)",
        "what": "A Ministry of Labour & Employment platform providing employment and career-related services including job search, career information, counselling and related opportunities.",
        "who": "Students, job seekers, employers and individuals seeking career guidance or employment-related services.",
        "how": "Create/login to an NCS profile, explore job and career services, and follow the official portal's current registration and application procedures.",
        "url": "https://www.ncs.gov.in/",
        # Self-referential edit: the source note also said this "doesn't
        # duplicate LearnMate AI's own Job Search feature" - LearnMate
        # Analytics AI has no Job Search feature, so that clause was
        # dropped rather than carried over as a false claim about this app.
        "note": "This links out to the official Government NCS ecosystem.",
    },
    {
        "icon": "\U0001F3E6",
        "title": "Vidya Lakshmi Portal",
        "what": "An education-loan portal designed to help students access information and apply for eligible education-loan opportunities through participating banks.",
        "who": "Students seeking education-financing options, subject to lender, course, institution and scheme eligibility.",
        "how": "Create/login to the official portal, review available education-loan options, compare applicable requirements and follow the official application process.",
        "url": "https://www.vidyalakshmi.co.in/",
        "note": "This is not financial advice, and eligibility or loan approval is never guaranteed.",
    },
]


def _render_service_card(service: dict) -> None:
    st.markdown(
        f"""<div class="lm-card">
        <h4 style="margin-top:0;">{service['icon']} {service['title']}</h4>
        <p><b>What it is</b><br/>{service['what']}</p>
        <p><b>Who it may be for</b><br/>{service['who']}</p>
        <p><b>How to access it</b><br/>{service['how']}</p>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("**Official website**")
    url = service.get("url")
    if url:
        st.link_button("\U0001F517 Visit Official Portal", url, use_container_width=True)
    else:
        st.caption("\u26A0\uFE0F Official portal link pending verification.")

    note = service.get("note")
    if note:
        st.caption(f"\u2139\uFE0F {note}")

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)


def render_government_services_page() -> None:
    st.markdown(
        '<span class="lm-badge">GOVERNMENT & STUDENT SERVICES</span>',
        unsafe_allow_html=True,
    )
    st.title("\U0001F1EE\U0001F1F3 Government & Student Support Services")
    st.caption(
        "A directory of official Government of India portals relevant to students - "
        "internships, skilling, scholarships, digital documents, academic identity, "
        "apprenticeships, career services, and education loans - each linking "
        "directly to the real official website."
    )

    st.info(
        "This section provides general guidance only. LearnMate Analytics AI is not "
        "an official representative of the Government of India and cannot issue "
        "any ID, card, certificate, or government document. Always verify "
        "current details on the official portal before taking action - "
        "eligibility, documents, and procedures may change.",
        icon="\u2139\uFE0F",
    )
    st.caption(
        "\u2139\uFE0F Government schemes, eligibility rules, deadlines, documents and "
        "portal procedures can change. Always verify the latest information "
        "on the official portal before applying."
    )

    st.divider()

    # Rendered two-per-row via a fresh st.columns(2) call for every pair,
    # rather than one `cols` object reused via `i % 2` - that reused-object
    # pattern stacks "row 2" inside whichever column its "row 1" sibling
    # landed in, so a taller card in one column can silently push that
    # column's next card out of line with its row neighbours.
    for row_start in range(0, len(GOVERNMENT_SERVICES), 2):
        row_services = GOVERNMENT_SERVICES[row_start:row_start + 2]
        row_cols = st.columns(2)
        for col, service in zip(row_cols, row_services):
            with col:
                _render_service_card(service)
