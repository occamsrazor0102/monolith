"""Shared fixtures and sample texts for pipeline tests."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AI_TEXT = (
    "This comprehensive analysis provides a thorough examination of the "
    "key factors that contribute to the overall effectiveness of the proposed "
    "framework. Furthermore, it is essential to note that the implementation "
    "of these strategies ensures alignment with best practices and industry "
    "standards. To address this challenge, we must consider multiple perspectives "
    "and leverage data-driven insights to achieve optimal outcomes. Additionally, "
    "this approach demonstrates the critical importance of systematic evaluation "
    "and evidence-based decision making in the modern landscape."
)

HUMAN_TEXT = (
    "so yeah I just kinda threw together a quick script to parse the logs "
    "and honestly it's pretty janky but it works lol. the main thing was "
    "getting the regex right for the timestamps because some of them had "
    "weird formats and I kept hitting edge cases. anyway I pushed it to the "
    "repo if you wanna take a look, but fair warning it's not exactly "
    "production ready haha. oh and I forgot to mention, there's a bug where "
    "it chokes on empty lines but I'll fix that tomorrow probably."
)

CLINICAL_TEXT = (
    "The patient presented to the emergency department with acute chest pain "
    "radiating to the left arm. Vital signs were stable with blood pressure "
    "of 130/85 mmHg and heart rate of 92 beats per minute. An electrocardiogram "
    "was performed which showed ST-segment elevation in leads V1 through V4. "
    "The patient was immediately started on aspirin and heparin therapy."
)
