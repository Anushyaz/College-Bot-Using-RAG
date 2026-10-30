"""
Sample College Regulations PDF Generator
Generates a realistic, structured multi-page PDF document containing official college
regulations, attendance rules, examination guidelines, fees, leave policies, and code of conduct.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


class NumberedCanvas:
    """Canvas that performs a two-pass rendering to add 'Page X of Y' footer."""
    def __init__(self, *args, **kwargs):
        pass


def add_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#64748b'))
    
    # Header
    canvas.drawString(54, 755, "Apex Institute of Higher Education & Technology | Academic Handbook 2026-2027")
    canvas.setStrokeColor(colors.HexColor('#cbd5e1'))
    canvas.setLineWidth(0.5)
    canvas.line(54, 750, 558, 750)
    
    # Footer
    canvas.line(54, 45, 558, 45)
    page_text = f"Official Policy Document - Page {doc.page}"
    canvas.drawString(54, 32, "Confidential & Proprietary - Office of the Academic Registrar")
    canvas.drawRightString(558, 32, page_text)
    canvas.restoreState()


def generate_sample_pdf(output_path="data/sample_college_regulations.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#0d9488'),
        spaceAfter=14
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'BulletDark',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    notice_style = ParagraphStyle(
        'NoticeBox',
        parent=body_style,
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#0f766e'),
        backColor=colors.HexColor('#f0fdfa'),
        borderColor=colors.HexColor('#99f6e4'),
        borderWidth=1,
        borderPadding=6,
        spaceBefore=6,
        spaceAfter=10
    )

    story = []
    
    # ==================== PAGE 1 ====================
    story.append(Paragraph("APEX INSTITUTE OF HIGHER EDUCATION & TECHNOLOGY", title_style))
    story.append(Paragraph("ACADEMIC REGULATIONS, EXAMINATION GUIDELINES & STUDENT CODE (2026-2027)", subtitle_style))
    story.append(Paragraph(
        "<b>Notice of Authority:</b> This manual governs all undergraduate, postgraduate, and diploma programs enrolled at Apex Institute. "
        "Every registered student is required to abide by these stipulations.",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    # Section 1
    story.append(Paragraph("ARTICLE 1: ACADEMIC CALENDAR & ENROLLMENT PROCEDURES", h1_style))
    story.append(Paragraph("<b>1.1 Semester Term Schedules:</b> The academic year consists of two primary semesters and an optional summer term:", body_style))
    story.append(Paragraph("&bull; <b>Fall Semester:</b> Commences on the first Monday of August and concludes mid-December (18 weeks duration).", bullet_style))
    story.append(Paragraph("&bull; <b>Spring Semester:</b> Commences on the second Monday of January and concludes mid-May (18 weeks duration).", bullet_style))
    story.append(Paragraph("&bull; <b>Summer Term:</b> Intensive 6-week remedial term running from June 1 to mid-July.", bullet_style))
    
    story.append(Paragraph("<b>1.2 Course Registration Deadlines:</b>", h2_style))
    story.append(Paragraph("Students must register for their courses within the designated window. The official Course Add/Drop deadline expires at 5:00 PM on the 10th academic day (end of Week 2) of each semester. Courses dropped during this period leave no record on the permanent transcript.", body_style))
    story.append(Paragraph("Course Withdrawal is permitted until the conclusion of Week 8. Withdrawn courses are recorded with a grade of 'W' on the transcript, which carries no grade-point penalty but does not entitle the student to any tuition refund.", body_style))
    
    # Section 2
    story.append(Paragraph("ARTICLE 2: ATTENDANCE REGULATIONS & CONDONATION RULES", h1_style))
    story.append(Paragraph("<b>2.1 Mandatory Minimum Attendance:</b> All registered students must maintain a minimum attendance of <b>75%</b> in all scheduled lectures, tutorials, practicals, and workshop sessions in each course independently.", body_style))
    
    story.append(Paragraph("<b>2.2 Condonation for Shortfall (65% to 74.9%):</b>", h2_style))
    story.append(Paragraph("Condonation of attendance deficiency between 65% and 74.9% may be sanctioned exclusively on substantiated medical grounds or university-authorized representation in inter-collegiate sports, academic symposiums, or national events.", body_style))
    story.append(Paragraph("&bull; <b>Procedure:</b> The student must submit formal supporting documents (hospital admission slips, physician certificates, event deputation orders) to the Head of Department within five (5) working days of return to campus.", bullet_style))
    story.append(Paragraph("&bull; <b>Condonation Fee:</b> A mandatory non-refundable condonation processing fee of <b>$50 per course</b> must be paid to the Accounts Branch upon approval.", bullet_style))
    story.append(Paragraph("&bull; <b>Sanctioning Authority:</b> Condonation can only be granted upon the joint recommendation of the Head of Department and formal approval by the Academic Dean.", bullet_style))

    story.append(Paragraph("<b>2.3 Severe Shortage and Course Detention (Below 65%):</b>", h2_style))
    story.append(Paragraph("Students whose attendance drops below <b>65%</b> in any registered subject are strictly disqualified and detained from taking the End-Semester Examination for that course. No condonation is legally permissible under any circumstances below 65%.", body_style))
    story.append(Paragraph("The student receives a grade of 'FA' (Failure due to Attendance) and is obligated to re-register for the course and attend regular classes during a subsequent semester, with full payment of the per-credit registration fee.", notice_style))
    
    story.append(PageBreak())
    
    # ==================== PAGE 2 ====================
    # Section 3
    story.append(Paragraph("ARTICLE 3: EXAMINATION POLICIES, GRADING SYSTEM & RE-EXAMINATION", h1_style))
    story.append(Paragraph("<b>3.1 Grading Scale and GPA Computation:</b> Student academic performance is graded on a 10-point scale based on continuous internal evaluation (40%) and end-semester examinations (60%):", body_style))
    
    table_data = [
        ["Grade", "Grade Points", "Percentage Range", "Qualitative Performance"],
        ["A+", "10.0", "90% - 100%", "Outstanding"],
        ["A", "9.0", "80% - 89%", "Excellent"],
        ["B+", "8.0", "70% - 79%", "Very Good"],
        ["B", "7.0", "60% - 69%", "Good"],
        ["C", "6.0", "50% - 59%", "Satisfactory / Average"],
        ["D", "5.0", "40% - 49%", "Pass"],
        ["F", "0.0", "Below 40%", "Fail (Re-examination required)"]
    ]
    t = Table(table_data, colWidths=[55, 75, 110, 160])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8.5),
        ('BOTTOMPADDING', (0,0), (-1,0), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('ALIGN', (0,0), (2,-1), 'CENTER'),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    story.append(t)
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("<b>3.2 What Happens If a Student Fails a Subject?</b>", h2_style))
    story.append(Paragraph("A student securing an 'F' grade in any course earns zero credit points for that course. A minimum cumulative grade point average (CGPA) of <b>2.0 on a 4.0 scale (or 5.0 on a 10.0 scale)</b> is required for degree award. A failed core subject must be cleared either through the Supplementary Re-Examination or by re-registering for the course.", body_style))
    
    story.append(Paragraph("<b>3.3 Rules for Applying for a Re-Examination (Supplementary Exam):</b>", h2_style))
    story.append(Paragraph("&bull; <b>Eligibility:</b> Students with an 'F' grade or students unable to attend the main exam due to documented medical emergencies are eligible.", bullet_style))
    story.append(Paragraph("&bull; <b>Application Window:</b> Formal application must be filed online through the Student Portal within <b>30 calendar days</b> of semester result publication.", bullet_style))
    story.append(Paragraph("&bull; <b>Re-Examination Fee:</b> A non-refundable fee of <b>$35 per subject</b> is applicable.", bullet_style))
    story.append(Paragraph("&bull; <b>Maximum Attempts:</b> A student is allowed a maximum of <b>three (3) attempts</b> to clear any given course. If unsuccessful after 3 attempts, the student must petition the Academic Council for an exceptional repeat semester.", bullet_style))
    
    story.append(Paragraph("<b>3.4 Re-Evaluation and Verification of Answer Scripts:</b>", h2_style))
    story.append(Paragraph("Students seeking verification or re-evaluation of answer scripts may apply within <b>14 days</b> of result notification. A scrutiny fee of <b>$25 per paper</b> applies. If the re-evaluation results in a grade increase of more than 10%, the fee is refunded.", body_style))

    # Section 4
    story.append(Paragraph("ARTICLE 4: LEAVE OF ABSENCE & MEDICAL POLICIES", h1_style))
    story.append(Paragraph("<b>4.1 How Can a Student Apply for a Leave of Absence?</b>", h2_style))
    story.append(Paragraph("Students may apply for a temporary formal Leave of Absence (LOA) for a maximum duration of <b>two (2) consecutive semesters</b>. Valid grounds include certified medical treatment, compulsory military/civil service, or verified entrepreneurial/research fellowship opportunities.", body_style))
    story.append(Paragraph("&bull; <b>Application Step 1:</b> Download Form LOA-101 from the registrar portal.", bullet_style))
    story.append(Paragraph("&bull; <b>Application Step 2:</b> Obtain written endorsement from Academic Proctor and Head of Department.", bullet_style))
    story.append(Paragraph("&bull; <b>Application Step 3:</b> Submit completed form along with third-party verification to the Office of the Registrar at least 15 days before the semester starts.", bullet_style))
    story.append(Paragraph("&bull; <b>Readmission:</b> Upon return, submit Form LOA-201 (Fitness/Resumption Notice) to reactivate student status without re-application fees.", bullet_style))
    
    story.append(Paragraph("<b>4.2 Routine Short Medical Leaves:</b>", h2_style))
    story.append(Paragraph("For short-term illness (1-10 days), a medical certificate signed by a licensed practitioner must be delivered to the Proctor within <b>5 working days</b> of returning to class. Medical absences exceeding 10 consecutive class days must be countersigned by the Chief Medical Officer at the Apex University Health Center.", body_style))

    story.append(PageBreak())

    # ==================== PAGE 3 ====================
    # Section 5
    story.append(Paragraph("ARTICLE 5: TUITION FEES, PAYMENT DEADLINES & SCHOLARSHIP GUIDELINES", h1_style))
    story.append(Paragraph("<b>5.1 Tuition Payment Schedule:</b>", h2_style))
    story.append(Paragraph("Tuition and institutional fees must be settled in full by the <b>15th calendar day</b> of the month in which the semester commences (August 15 for Fall; January 15 for Spring).", body_style))
    story.append(Paragraph("&bull; <b>Grace Period:</b> A grace period of 7 calendar days is permitted without penalty.", bullet_style))
    story.append(Paragraph("&bull; <b>Late Payment Penalty:</b> Commencing day 8 following the deadline, a late fee of <b>$10 per business day</b> is charged up to a maximum of 21 calendar days.", bullet_style))
    story.append(Paragraph("&bull; <b>Suspension of Access:</b> If fees remain unpaid after 21 calendar days, student portal access, laboratory credentials, and semester exam hall tickets are suspended until the account is cleared.", bullet_style))
    
    story.append(Paragraph("<b>5.2 Institutional Merit Scholarship Criteria:</b>", h2_style))
    story.append(Paragraph("The Apex Dean's Merit Scholarship awards a 30% to 50% tuition waiver to qualifying scholars. Eligibility criteria include:", body_style))
    story.append(Paragraph("&bull; Maintaining a minimum cumulative GPA (CGPA) of <b>3.75 out of 4.0 (or 8.75 out of 10.0)</b>.", bullet_style))
    story.append(Paragraph("&bull; Registering for a full academic load (minimum 15 credits per semester).", bullet_style))
    story.append(Paragraph("&bull; No recorded disciplinary warnings or integrity violations on the student file.", bullet_style))

    # Section 6
    story.append(Paragraph("ARTICLE 6: TRANSCRIPTS, GRADE CARDS & OFFICIAL CERTIFICATES", h1_style))
    story.append(Paragraph("<b>6.1 What is the Procedure for Obtaining an Official Transcript?</b>", h2_style))
    story.append(Paragraph("Transcripts may be requested through the central Student Information System (SIS) under 'Document Services':", body_style))
    story.append(Paragraph("&bull; <b>Standard Processing:</b> Processed within <b>7 business days</b>. Fee: <b>$15 per sealed official hardcopy</b>.", bullet_style))
    story.append(Paragraph("&bull; <b>Tatkal (Expedited) Service:</b> Processed and dispatched within <b>48 hours</b>. Fee: <b>$30 per copy</b>.", bullet_style))
    story.append(Paragraph("&bull; <b>Digitally Signed PDF:</b> Digitally certified copy delivered via secure email within 3 business days. Fee: <b>$10 per e-copy</b>.", bullet_style))
    story.append(Paragraph("&bull; <b>Dispatch:</b> Domestic postal dispatch is included in the fee; overseas international courier requires an additional $45 shipping charge.", bullet_style))

    story.append(Paragraph("<b>6.2 Duplicate Identity Cards & Certificates:</b>", h2_style))
    story.append(Paragraph("Lost student identity smartcards are re-issued within 2 business days following submission of a lost-item declaration and payment of a <b>$20 replacement fee</b>.", body_style))

    # Section 7
    story.append(Paragraph("ARTICLE 7: STUDENT CODE OF CONDUCT & DISCIPLINARY REGULATIONS", h1_style))
    story.append(Paragraph("<b>7.1 Academic Integrity and Plagiarism:</b>", h2_style))
    story.append(Paragraph("Any academic assignment, laboratory report, or research thesis exhibiting a plagiarism similarity index exceeding <b>20%</b> will automatically be referred to the Academic Integrity Board.", body_style))
    story.append(Paragraph("&bull; <b>First Offense:</b> Zero marks on the assignment and a formal letter of reprimand.", bullet_style))
    story.append(Paragraph("&bull; <b>Second Offense:</b> Immediate award of grade 'F' in the course and disciplinary probation for one semester.", bullet_style))
    
    story.append(Paragraph("<b>7.2 Examination Hall Protocol & Prohibited Devices:</b>", h2_style))
    story.append(Paragraph("Mobile phones, smartwatch devices, electronic storage devices, and unauthorized formula sheets are strictly forbidden inside the examination hall. Any student discovered in possession of an unauthorized device—irrespective of whether it was active—faces immediate exam paper cancellation, confiscation of the device, and a minimum one-semester academic suspension.", body_style))

    # Section 8
    story.append(Spacer(1, 10))
    story.append(Paragraph("ARTICLE 8: OFFICIAL INTERPRETATION & DISCLAIMER", h1_style))
    story.append(Paragraph(
        "<b>Institutional Disclaimer:</b> The policies and figures outlined in this document represent the authoritative rules "
        "ratified by the Academic Council for academic year 2026-2027. However, the <b>Office of the Academic Registrar</b> remains "
        "the final authority for resolving ambiguities, grievances, and exceptional waivers. In cases of conflict with individual "
        "departmental notices, this Institute Handbook supersedes all subordinate documents.",
        notice_style
    ))

    # Build PDF
    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    print(f"Successfully generated sample college regulations PDF at: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_sample_pdf()
