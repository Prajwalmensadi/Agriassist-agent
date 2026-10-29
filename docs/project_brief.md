# AgriAssist — Project Brief

## Problem
Farmers may receive agricultural recommendations but still need to determine which actions fit their crop, growth stage, symptoms, resources, and current conditions. They may also need clarification and follow-up after taking action.

## Proposed solution
Build a bounded agent that collects relevant context, identifies missing information, retrieves guidance from a curated collection of verified agricultural sources, generates an evidence-grounded action plan, records farmer-reported actions, and tracks subsequent observations.

## Initial scope
- Supported crop: paddy (rice) only.
- Target region: coastal Karnataka, India, with local relevance verified during Week 1.
- Other crops are outside the current MVP scope and must not receive agricultural advice from AgriAssist. The architecture may be extended to other crops in a future version.
- Interface: English-first, with a limited Kannada interaction path to be evaluated.
- Local-first inference to respect a zero AI API budget.
- Persistent case memory in SQLite.
- A simple Streamlit interface by Week 5.

## Out of scope
- Autonomous pesticide prescriptions or invented application rates.
- Image-based diagnosis.
- Any crop other than paddy (rice) in the current MVP.
- Nationwide coverage beyond the initial coastal Karnataka target.
- Claims of proven field impact.

## Safety
AgriAssist is a decision-support prototype. It must communicate uncertainty, cite retrieved evidence, avoid unsupported treatment advice, and recommend qualified expert inspection when needed.

## Success
At least 15 predefined tests with recorded results; relevant retrieval; source-supported claims; valid structured outputs; correct case state and follow-up handling; bounded execution; logs; reproducible setup; and a five-minute demo.
