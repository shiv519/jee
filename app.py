# app.py
import streamlit as st
import PyPDF2
import io
import re
import time
from datetime import datetime
import base64
import pandas as pd
from transformers import pipeline
import torch

# Initialize NLP classifier
try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
except:
    classifier = None

# Categories for classification
CATEGORIES = ["Mathematics", "Physics", "Chemistry"]

def is_question(text):
    """Determine if text is likely a question (not solution/answer key)."""
    text = text.lower().strip()
    
    # Skip empty or very short text
    if len(text) < 20:
        return False
        
    # Skip sections that look like solutions/answers
    solution_indicators = ['solution:', 'answer key', 'explanation:', 'answers:', 
                          'correct option', 'key answers', 'sol.', 'ans.']
    if any(indicator in text for indicator in solution_indicators):
        return False
        
    # Skip lines that are just numbers or letters (likely answer keys)
    if re.match(r'^\s*\d+\.?\s*[a-d]\s*$', text, re.IGNORECASE):
        return False
        
    # Look for question patterns
    question_indicators = ['what', 'which', 'how many', 'calculate', 'determine', 
                          'find', 'prove', 'show that', 'ratio of', 'value of']
    if any(indicator in text for indicator in question_indicators):
        return True
        
    # Look for question numbers
    if re.match(r'^\s*(Q|Question)?\s*\d+[.)]', text[:20], re.IGNORECASE):
        return True
        
    return False

def extract_questions_from_pdf(pdf_file):
    """Extract text from PDF and split into questions."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    # Split text into paragraphs/lines
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    questions = []
    current_question = None
    question_number = 1
    
    for para in paragraphs:
        # Handle paragraph that starts a new question
        if is_question(para):
            # If we have a current question being built, save it first
            if current_question:
                questions.append(current_question)
                question_number += 1
                
            # Start new question
            current_question = {
                "number": str(question_number),
                "text": para,
                "subject": None,
                "options": [],
                "answer": None,
                "marked": False
            }
        elif current_question:
            # This might be options or continuation of current question
            # Check if it looks like options (A), (B) etc.
            option_match = re.match(r'^\s*\(([a-d])\)\s*(.*)$', para, re.IGNORECASE)
            if option_match:
                current_question['options'].append(option_match.group(2))
            else:
                # Otherwise append to question text
                current_question['text'] += "\n" + para
    
    # Add the last question if there is one
    if current_question:
        questions.append(current_question)
    
    return questions

def classify_question(question_text):
    """Classify question into subject using NLP."""
    if classifier:
        result = classifier(question_text, candidate_labels=CATEGORIES)
        return result['labels'][0]
    return None

def create_test_interface(questions):
    """Create the interactive test interface."""
    st.markdown("""
    <style>
        .subject-tab {
            transition: all 0.3s ease;
            margin-right: 8px;
            margin-bottom: 8px;
        }
        .subject-tab:hover {
            transform: translateY(-2px);
        }
        .question-card {
            padding: 16px;
            margin-bottom: 16px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .question-card:hover {
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .option-selected {
            background-color: #3b82f6 !important;
            color: white !important;
            border-color: #3b82f6 !important;
        }
        .option-btn {
            display: block;
            width: 100%;
            text-align: left;
            margin-bottom: 8px;
            padding: 8px 12px;
            border-radius: 4px;
            background-color: white;
            border: 1px solid #e2e8f0;
            cursor: pointer;
        }
        .option-btn:hover {
            background-color: #f8fafc;
        }
        .timer {
            font-family: monospace;
            font-size: 24px;
            color: #ef4444;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .pdf-preview {
            height: 500px;
            overflow-y: auto;
            border: 1px solid #e2e8f0;
            padding: 10px;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_subject' not in st.session_state:
        st.session_state.current_subject = "All Questions"
    
    if 'answers' not in st.session_state:
        st.session_state.answers = {q['number']: None for q in questions}
    
    if 'marked_questions' not in st.session_state:
        st.session_state.marked_questions = set()
    
    # Header with timer
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("JEE Question Analyzer")
    with col2:
        st.markdown('<div class="timer" id="timer">00:00:00</div>', unsafe_allow_html=True)
    
    # Subject tabs
    tab_cols = st.columns(len(CATEGORIES) + 1)
    tabs = ["All Questions"] + CATEGORIES
    for i, tab in enumerate(tabs):
        if tab_cols[i].button(tab, key=f"tab_{tab}"):
            st.session_state.current_subject = tab
    
    # Filter questions by subject
    if st.session_state.current_subject != "All Questions":
        filtered_questions = [q for q in questions if q.get('subject') == st.session_state.current_subject]
    else:
        filtered_questions = questions
    
    if not questions:
        st.warning("No questions were found in the PDF. Please try a different file.")
        return
    
    # Question palette and display
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Question Palette")
        st.caption("Green: Answered, Yellow: Marked, Red: Unanswered")
        
        palette_cols = 5
        rows = (len(questions) + palette_cols - 1) // palette_cols
        for row in range(rows):
            cols = st.columns(palette_cols)
            for i in range(palette_cols):
                idx = row * palette_cols + i
                if idx < len(questions):
                    q_num = questions[idx]['number']
                    status = "🟢" if st.session_state.answers.get(q_num) else "🔴"
                    if q_num in st.session_state.marked_questions:
                        status = "🟡"
                    if cols[i].button(f"{status} {q_num}", key=f"palette_{q_num}"):
                        st.session_state.current_question = idx
    
    with col2:
        if 'current_question' not in st.session_state:
            st.session_state.current_question = 0
        
        current_q = questions[st.session_state.current_question]
        
        st.subheader(f"Question {current_q['number']}")
        st.markdown(f"<div style='margin-bottom: 20px;'>{current_q['text']}</div>", unsafe_allow_html=True)
        
        # Display options if available
        if current_q.get('options'):
            st.write("**Options:**")
            for idx, option in enumerate(current_q['options']):
                is_selected = st.session_state.answers.get(current_q['number']) == idx
                btn_class = "option-selected" if is_selected else ""
                option_html = f"""
                <button class="option-btn {btn_class}" 
                        onclick="parent.document.getElementById('select-option-{idx}').click()">
                    {chr(65 + idx)}. {option}
                </button>
                """
                st.markdown(option_html, unsafe_allow_html=True)
                
                if st.button(f"Select Option {chr(65 + idx)}", 
                           key=f"select-option-{idx}", 
                           on_click=select_option, 
                           args=(current_q['number'], idx),
                           visible=False):
                    pass
        
        # Mark question button
        col1, col2, col3 = st.columns([2, 1, 1])
        is_marked = current_q['number'] in st.session_state.marked_questions
        mark_label = "✅ Unmark Question" if is_marked else "🔖 Mark for Review"
        if col1.button(mark_label):
            if is_marked:
                st.session_state.marked_questions.remove(current_q['number'])
            else:
                st.session_state.marked_questions.add(current_q['number'])
            st.experimental_rerun()
        
        # Navigation buttons
        if col2.button("⏮ Previous") and st.session_state.current_question > 0:
            st.session_state.current_question -= 1
            st.experimental_rerun()
        
        if col3.button("Next ⏭") and st.session_state.current_question < len(questions)-1:
            st.session_state.current_question += 1
            st.experimental_rerun()

def select_option(question_num, option_idx):
    """Callback for when an option is selected."""
    st.session_state.answers[question_num] = option_idx
    st.experimental_rerun()

def main():
    st.set_page_config(page_title="JEE Question Analyzer", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
        .stFileUploader > div:first-child {
            width: 100%;
        }
        .stFileUploader > div:first-child > div:first-child {
            display: none;
        }
        .upload-area {
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
        }
        .upload-area:hover {
            border-color: #3b82f6;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.title("JEE Question Analyzer")
    st.write("Upload a PDF of questions to analyze them by subject")
    
    with st.container():
        st.markdown('<div class="upload-area">Drag and drop PDF here or click to browse</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="hidden")
    
    if uploaded_file is not None:
        # Display PDF preview
        st.subheader("PDF Preview")
        with st.expander("View Uploaded PDF"):
            base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
            pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf">'
            st.markdown(pdf_display, unsafe_allow_html=True)
            uploaded_file.seek(0)  # Reset file pointer
        
        # Process PDF and classify questions
        with st.spinner("Analyzing questions..."):
            questions = extract_questions_from_pdf(uploaded_file)
            
            for question in questions:
                question['subject'] = classify_question(question['text'])
                
                # Add mock options if none found
                if not question.get('options'):
                    question['options'] = [
                        f"Option A for question {question['number']}",
                        f"Option B for question {question['number']}",
                        f"Option C for question {question['number']}",
                        f"Option D for question {question['number']}"
                    ]
        
        # Start test interface
        create_test_interface(questions)

if __name__ == "__main__":
    main()
