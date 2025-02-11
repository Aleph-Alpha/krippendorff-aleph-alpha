import re


ORDINAL_CATEGORIES = [
    # Sentiment & Opinion Scales
    ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"],
    ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
    ["Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"],
    # Ratings & Reviews
    ["Poor", "Fair", "Good", "Very Good", "Excellent"],
    ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"],
    ["Unusable", "Difficult", "Neutral", "Easy", "Very Easy"],
    # Performance & Proficiency Levels
    ["Beginner", "Intermediate", "Advanced", "Expert"],
    ["F", "D", "C", "B", "A"],  # Grading scale
    ["Novice", "Basic", "Conversational", "Fluent", "Native"],
    # Risk & Severity Levels
    ["No Pain", "Mild", "Moderate", "Severe", "Extreme"],  # Pain scale
    ["Very Low", "Low", "Moderate", "High", "Very High"],  # Risk assessment
    ["Not Urgent", "Slightly Urgent", "Moderately Urgent", "Urgent", "Critical"],
    # Likelihood & Probability
    ["Impossible", "Unlikely", "Neutral", "Likely", "Certain"],
    ["Not Confident", "Slightly Confident", "Moderately Confident", "Very Confident", "Completely Confident"],
    # Custom/Other Scales
    ["Terrible", "Bad", "Okay", "Good", "Excellent"],  # Alternative sentiment scale
    ["Very Low", "Low", "Medium", "High", "Very High"],  # Generic five-level scale
]


TEXT_COLUMN_ALIASES = {
    "text",
    "Text",
    "sentence",
    "Sentence",
    "Paragraph",
    "Document",
    "paragraph",
    "document",
    "texts",
    "sentences",
    "documents",
    "paragraphs",
}

WORD_COLUMN_ALIASES = {"word", "Word", "token", "Token", "tokens", "words"}

ANNOTATOR_REGEX = re.compile(r"^(annotator[_\s]*\d+|Annotator[_\s]*\d+|annotator one|Annotator one)$", re.IGNORECASE)
