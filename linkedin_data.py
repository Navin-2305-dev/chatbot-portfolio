from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

LINKEDIN_PROFILE_PATH = os.getenv("LINKEDIN_PROFILE_PATH", "linkedin_profile.json")


# ── Loader ────────────────────────────────────────────────────────────────────

def load_linkedin_profile() -> dict:
    """
    Load linkedin_profile.json from disk.
    Writes a minimal placeholder and raises a clear error if missing,
    so the build script fails loudly rather than silently.
    """
    path = Path(LINKEDIN_PROFILE_PATH)

    if not path.exists():
        placeholder = {"name": "Navin B", "headline": "", "summary": ""}
        path.write_text(json.dumps(placeholder, indent=2), encoding="utf-8")
        raise FileNotFoundError(
            f"linkedin_profile.json not found at '{path.resolve()}'.\n"
            "A blank placeholder has been written — please fill it in.\n"
            "Tip: run  python linkedin_data.py --from-export <dir>  "
            "to auto-populate from an official LinkedIn data export."
        )

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"LinkedIn profile loaded: '{path}'")
    return data


# ── Section builders ──────────────────────────────────────────────────────────

def _build_profile_header(profile: dict) -> Document:
    """
    Name, headline, contact details, location, open-to-work status.
    This is what gets retrieved for greetings and identity questions.
    """
    lines = [
        f"Name: {profile.get('name', '')}",
        f"Professional Headline: {profile.get('headline', '')}",
        f"Location: {profile.get('location', '')}",
        f"LinkedIn: {profile.get('profile_url', '')}",
        f"Email: {profile.get('email', '')}",
        f"Phone: {profile.get('phone', '')}",
        f"Address: {profile.get('address', '')}",
    ]

    if profile.get("open_to_work"):
        looking = profile.get("looking_for", "new opportunities")
        lines.append(f"Open to Work: Yes — looking for {looking}")

    current_status = profile.get("current_status", "")
    if current_status:
        lines.append(f"Current Status: {current_status}")

    return Document(
        page_content="\n".join(lines),
        metadata={"source": "linkedin", "section": "profile_header"},
    )


def _build_summary(profile: dict) -> List[Document]:
    """
    Full About / bio text as its own Document.
    Kept separate from the header so long summaries don't dilute
    identity-focused retrieval.
    """
    summary = profile.get("summary", "").strip()
    if not summary:
        return []

    return [Document(
        page_content=f"Professional Summary / About:\n\n{summary}",
        metadata={"source": "linkedin", "section": "summary"},
    )]


def _build_current_status(profile: dict) -> List[Document]:
    """
    Quick-access snapshot for queries like 'what are you doing now?'
    Combines current_status field, open-to-work, and latest experience.
    """
    parts: List[str] = ["Current Status Snapshot:"]

    status = profile.get("current_status", "")
    if status:
        parts.append(status)

    if profile.get("open_to_work"):
        looking = profile.get("looking_for", "new opportunities")
        parts.append(f"Actively looking for: {looking}")

    # Pull the most recent experience entry
    experiences = profile.get("experience", [])
    if experiences:
        latest = experiences[0]
        parts.append(
            f"Latest Role: {latest.get('title', '')} at {latest.get('company', '')} "
            f"({latest.get('start_date', '')} — {latest.get('end_date', '')})"
        )

    if len(parts) == 1:
        return []

    return [Document(
        page_content="\n".join(parts),
        metadata={"source": "linkedin", "section": "current_status"},
    )]


def _build_experience(profile: dict) -> List[Document]:
    """
    One Document per role for fine-grained retrieval.
    A user asking 'what did you do at IBM?' gets exactly that document.
    """
    docs: List[Document] = []
    experiences = profile.get("experience", [])

    if not experiences:
        return docs

    # Also produce one consolidated experience overview doc
    overview_lines = [
        f"Work Experience Overview ({len(experiences)} roles):",
    ]
    for exp in experiences:
        overview_lines.append(
            f"  • {exp.get('title', '')} at {exp.get('company', '')} "
            f"({exp.get('start_date', '')} — {exp.get('end_date', '')})"
        )

    docs.append(Document(
        page_content="\n".join(overview_lines),
        metadata={"source": "linkedin", "section": "experience_overview"},
    ))

    # One detailed Document per role
    for exp in experiences:
        title     = exp.get("title", "")
        company   = exp.get("company", "")
        location  = exp.get("location", "")
        start     = exp.get("start_date", "")
        end       = exp.get("end_date", "")
        desc      = exp.get("description", "")

        content = (
            f"Role: {title}\n"
            f"Company: {company}\n"
            f"Location: {location}\n"
            f"Duration: {start} — {end}\n"
        )
        if desc:
            content += f"Description: {desc}"

        docs.append(Document(
            page_content=content,
            metadata={
                "source": "linkedin",
                "section": "experience",
                "company": company,
                "title": title,
            },
        ))

    return docs


def _build_education(profile: dict) -> List[Document]:
    """
    Education entries. Handles both old schema (graduation_year only)
    and new schema (start_year + graduation_year + honors).
    """
    education = profile.get("education", [])
    if not education:
        return []

    blocks = ["Education:"]
    for edu in education:
        degree      = edu.get("degree", "")
        institution = edu.get("institution", "")
        start       = edu.get("start_year", "")
        end         = edu.get("graduation_year", "")
        grade       = edu.get("grade", "")
        honors      = edu.get("honors", "")

        duration = f"{start} — {end}" if start else end

        lines = [
            f"\nDegree: {degree}",
            f"Institution: {institution}",
        ]
        if duration:
            lines.append(f"Duration: {duration}")
        if grade:
            lines.append(f"Grade / CGPA: {grade}")
        if honors:
            lines.append(f"Honors: {honors}")

        blocks.append("\n".join(lines))

    return [Document(
        page_content="\n".join(blocks),
        metadata={"source": "linkedin", "section": "education"},
    )]


def _build_skills(profile: dict) -> List[Document]:
    """
    All skills in one Document, with top skills highlighted at the top
    so they appear in every skills-related retrieval result.
    """
    skills     = profile.get("skills", [])
    top_skills = profile.get("top_skills", [])

    if not skills:
        return []

    lines = ["Technical Skills and Competencies:"]

    if top_skills:
        lines.append(f"Top Skills: {', '.join(top_skills)}")

    lines.append(f"All Skills: {', '.join(skills)}")

    return [Document(
        page_content="\n".join(lines),
        metadata={"source": "linkedin", "section": "skills"},
    )]


def _build_certifications(profile: dict) -> List[Document]:
    """
    Certifications. Shows name, issuer, and date where available.
    """
    certs = profile.get("certifications", [])
    if not certs:
        return []

    blocks = [f"Certifications ({len(certs)} total):"]

    for cert in certs:
        name   = cert.get("name", "")
        issuer = cert.get("issuer", "")
        date   = cert.get("date", "")

        line = f"\n• {name}"
        if issuer:
            line += f"\n  Issued by: {issuer}"
        if date:
            line += f"\n  Date: {date}"
        blocks.append(line)

    return [Document(
        page_content="\n".join(blocks),
        metadata={"source": "linkedin", "section": "certifications"},
    )]


def _build_hackathons(profile: dict) -> List[Document]:
    """
    Hackathon wins as a dedicated Document.
    Handles the total_hackathon_wins count and individual entries.
    Answers queries like 'how many hackathons have you won?' precisely.
    """
    hackathons = profile.get("hackathons", [])
    total      = profile.get("total_hackathon_wins", len(hackathons))

    if not hackathons and not total:
        return []

    lines = [f"Hackathon Achievements — {total} Wins:"]

    for h in hackathons:
        name   = h.get("name", "")
        result = h.get("result", "Winner")
        lines.append(f"  • {name}: {result}")

    return [Document(
        page_content="\n".join(lines),
        metadata={
            "source": "linkedin",
            "section": "hackathons",
            "total_wins": total,
        },
    )]


def _build_notable_projects(profile: dict) -> List[Document]:
    """
    AI / portfolio projects mentioned in the LinkedIn profile.
    One overview doc + one doc per project for targeted retrieval.
    """
    projects = profile.get("notable_projects", [])
    if not projects:
        return []

    docs: List[Document] = []

    # Overview
    overview_lines = [f"Notable Projects ({len(projects)}):"]
    for p in projects:
        overview_lines.append(f"  • {p.get('name', '')}: {p.get('description', '')}")

    docs.append(Document(
        page_content="\n".join(overview_lines),
        metadata={"source": "linkedin", "section": "projects_overview"},
    ))

    # One doc per project
    for project in projects:
        name = project.get("name", "")
        desc = project.get("description", "")
        tech = project.get("tech", "")

        content = f"Project: {name}\nDescription: {desc}"
        if tech:
            content += f"\nTech Used: {tech}"

        docs.append(Document(
            page_content=content,
            metadata={
                "source": "linkedin",
                "section": "project",
                "project_name": name,
            },
        ))

    return docs


def _build_extras(profile: dict) -> List[Document]:
    """
    Languages spoken, honors/awards, publications.
    """
    languages    = profile.get("languages", [])
    honors       = profile.get("honors", [])
    publications = profile.get("publications", [])

    parts: List[str] = []

    if languages:
        parts.append(f"Languages Spoken: {', '.join(languages)}")

    if honors:
        honor_lines = ["Honors & Awards:"] + [f"  • {h}" for h in honors]
        parts.append("\n".join(honor_lines))

    if publications:
        pub_lines = ["Publications:"] + [f"  • {p}" for p in publications]
        parts.append("\n".join(pub_lines))

    if not parts:
        return []

    return [Document(
        page_content="\n\n".join(parts),
        metadata={"source": "linkedin", "section": "extras"},
    )]


# ── Public API ────────────────────────────────────────────────────────────────

def linkedin_profile_to_documents() -> List[Document]:
    """
    Load linkedin_profile.json and return a list of LangChain Documents,
    one per logical section, ready to embed and insert into Qdrant.
    """
    profile = load_linkedin_profile()
    docs: List[Document] = []

    docs.append(_build_profile_header(profile))
    docs.extend(_build_summary(profile))
    docs.extend(_build_current_status(profile))
    docs.extend(_build_experience(profile))
    docs.extend(_build_education(profile))
    docs.extend(_build_skills(profile))
    docs.extend(_build_certifications(profile))
    docs.extend(_build_hackathons(profile))
    docs.extend(_build_notable_projects(profile))
    docs.extend(_build_extras(profile))

    logger.info(f"LinkedIn → {len(docs)} documents generated")
    return docs


# ── CLI: build JSON from official LinkedIn CSV export ─────────────────────────

def build_from_linkedin_export(export_dir: str) -> None:
    """
    Parse official LinkedIn data export CSVs into linkedin_profile.json.

    HOW TO GET YOUR LINKEDIN EXPORT:
      1. LinkedIn → Me → Settings & Privacy → Data Privacy
      2. → Get a copy of your data → Request archive
      3. Unzip the downloaded file
      4. Run: python linkedin_data.py --from-export /path/to/unzipped/

    Populates: name, headline, summary, location, experience,
               education, skills, certifications.
    Fields like hackathons, notable_projects, phone, email must be
    added manually to the generated JSON afterward.
    """
    import csv

    base    = Path(export_dir)
    profile: dict = {}

    # Profile.csv — basic identity
    profile_csv = base / "Profile.csv"
    if profile_csv.exists():
        with open(profile_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                profile["name"]        = f"{row.get('First Name','')} {row.get('Last Name','')}".strip()
                profile["headline"]    = row.get("Headline", "")
                profile["summary"]     = row.get("Summary", "")
                profile["location"]    = row.get("Geo Location", "")
                profile["profile_url"] = row.get("Public Profile Url", "")
                profile["email"]       = row.get("Email Address", "")
                break
        logger.info("Profile.csv parsed")
    else:
        logger.warning(f"Profile.csv not found in {export_dir}")

    # Positions.csv — work experience
    positions_csv = base / "Positions.csv"
    if positions_csv.exists():
        experiences = []
        with open(positions_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                experiences.append({
                    "title":       row.get("Title", ""),
                    "company":     row.get("Company Name", ""),
                    "location":    row.get("Location", ""),
                    "start_date":  row.get("Started On", ""),
                    "end_date":    row.get("Finished On", "") or "Present",
                    "description": row.get("Description", ""),
                })
        profile["experience"] = experiences
        logger.info(f"Positions.csv parsed — {len(experiences)} roles")

    # Education.csv
    education_csv = base / "Education.csv"
    if education_csv.exists():
        education = []
        with open(education_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                education.append({
                    "degree":          f"{row.get('Degree Name','')} {row.get('Field Of Study','')}".strip(),
                    "institution":     row.get("School Name", ""),
                    "start_year":      row.get("Start Date", ""),
                    "graduation_year": row.get("End Date", ""),
                    "grade":           row.get("Grade", ""),
                    "honors":          row.get("Activities", ""),
                })
        profile["education"] = education
        logger.info(f"Education.csv parsed — {len(education)} entries")

    # Skills.csv
    skills_csv = base / "Skills.csv"
    if skills_csv.exists():
        skills = []
        with open(skills_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                skill = row.get("Name", "").strip()
                if skill:
                    skills.append(skill)
        profile["skills"] = skills
        logger.info(f"Skills.csv parsed — {len(skills)} skills")

    # Certifications.csv
    certs_csv = base / "Certifications.csv"
    if certs_csv.exists():
        certs = []
        with open(certs_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                certs.append({
                    "name":   row.get("Name", ""),
                    "issuer": row.get("Authority", ""),
                    "date":   row.get("Started On", ""),
                })
        profile["certifications"] = certs
        logger.info(f"Certifications.csv parsed — {len(certs)} certs")

    # Stub fields that must be filled manually
    profile.setdefault("phone", "")
    profile.setdefault("address", "")
    profile.setdefault("open_to_work", True)
    profile.setdefault("looking_for", "")
    profile.setdefault("top_skills", [])
    profile.setdefault("hackathons", [])
    profile.setdefault("total_hackathon_wins", 0)
    profile.setdefault("notable_projects", [])
    profile.setdefault("languages", [])
    profile.setdefault("honors", [])
    profile.setdefault("publications", [])
    profile.setdefault("current_status", "")

    out_path = Path(LINKEDIN_PROFILE_PATH)
    out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"✅  linkedin_profile.json written → {out_path.resolve()}")
    print("   Remember to fill in: phone, address, hackathons, notable_projects, current_status")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LinkedIn data helper for Navin's portfolio chatbot"
    )
    parser.add_argument(
        "--from-export",
        metavar="DIR",
        help="Parse official LinkedIn CSV export folder → linkedin_profile.json",
    )
    args = parser.parse_args()

    if args.from_export:
        build_from_linkedin_export(args.from_export)
    else:
        # Quick smoke test — print all generated docs
        logging.basicConfig(level=logging.INFO)
        docs = linkedin_profile_to_documents()
        print(f"\nTotal documents: {len(docs)}\n{'=' * 60}")
        for doc in docs:
            section = doc.metadata.get("section", "unknown")
            print(f"\n[{section}]")
            print(doc.page_content)
        print("\n" + "=" * 60)