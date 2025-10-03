# app/services/disease_classifier.py
import re
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DiseaseCategory:
    """Enhanced disease category with medical taxonomy"""
    code: str
    name: str
    description: str
    keywords: List[str]
    icd10_codes: List[str]
    severity_indicators: List[str]
    seasonal_pattern: str
    typical_age_groups: List[str]
    contagious: bool
    chronic: bool

class EnhancedDiseaseClassifier:
    """
    Advanced disease classification system with medical taxonomy
    
    Features:
    - ICD-10 inspired categorization
    - Severity level detection
    - Seasonal pattern recognition
    - Age group correlation
    - Symptom-based classification
    """
    
    def __init__(self):
        self.disease_categories = self._initialize_disease_taxonomy()
        self.severity_keywords = self._initialize_severity_keywords()
        
    def _initialize_disease_taxonomy(self) -> Dict[str, DiseaseCategory]:
        """Initialize comprehensive disease taxonomy"""
        
        categories = {
            'respiratory_upper': DiseaseCategory(
                code='RESP_UPPER',
                name='Upper Respiratory Infections',
                description='Common cold, flu, sinusitis, pharyngitis',
                keywords=['cold', 'flu', 'cough', 'sore throat', 'runny nose', 'nasal congestion', 
                         'sinusitis', 'pharyngitis', 'laryngitis', 'rhinitis', 'sneezing'],
                icd10_codes=['J00-J06', 'J30-J39'],
                severity_indicators=['mild fever', 'low grade fever'],
                seasonal_pattern='winter_peak',
                typical_age_groups=['all_ages'],
                contagious=True,
                chronic=False
            ),
            
            'respiratory_lower': DiseaseCategory(
                code='RESP_LOWER',
                name='Lower Respiratory Infections',
                description='Pneumonia, bronchitis, bronchiolitis',
                keywords=['pneumonia', 'bronchitis', 'bronchiolitis', 'chest infection', 
                         'lung infection', 'difficulty breathing', 'shortness of breath',
                         'chest pain', 'productive cough', 'wheezing'],
                icd10_codes=['J12-J18', 'J20-J22', 'J40-J47'],
                severity_indicators=['high fever', 'severe', 'hospitalization'],
                seasonal_pattern='winter_peak',
                typical_age_groups=['elderly', 'children', 'immunocompromised'],
                contagious=True,
                chronic=False
            ),
            
            'gastrointestinal_acute': DiseaseCategory(
                code='GI_ACUTE',
                name='Acute Gastrointestinal Disorders',
                description='Food poisoning, gastroenteritis, acute diarrhea',
                keywords=['diarrhea', 'vomiting', 'nausea', 'stomach pain', 'abdominal pain',
                         'food poisoning', 'gastroenteritis', 'stomach flu', 'dehydration',
                         'loose stools', 'stomach cramps'],
                icd10_codes=['K59', 'A08', 'A09', 'K30'],
                severity_indicators=['severe dehydration', 'blood in stool'],
                seasonal_pattern='summer_peak',
                typical_age_groups=['all_ages'],
                contagious=True,
                chronic=False
            ),
            
            'gastrointestinal_chronic': DiseaseCategory(
                code='GI_CHRONIC',
                name='Chronic Gastrointestinal Disorders',
                description='IBS, IBD, GERD, chronic gastritis',
                keywords=['ibs', 'irritable bowel', 'crohn', 'ulcerative colitis', 'gerd',
                         'acid reflux', 'chronic gastritis', 'inflammatory bowel',
                         'chronic abdominal pain', 'chronic diarrhea'],
                icd10_codes=['K58', 'K50', 'K51', 'K21'],
                severity_indicators=['chronic', 'persistent', 'long-term'],
                seasonal_pattern='no_pattern',
                typical_age_groups=['adults', 'elderly'],
                contagious=False,
                chronic=True
            ),
            
            'infectious_viral': DiseaseCategory(
                code='INF_VIRAL',
                name='Viral Infections',
                description='Viral fever, viral syndrome, common viral infections',
                keywords=['viral fever', 'viral infection', 'viral syndrome', 'fever',
                         'body aches', 'fatigue', 'malaise', 'headache', 'chills'],
                icd10_codes=['B34', 'R50'],
                severity_indicators=['high fever', 'severe fatigue'],
                seasonal_pattern='seasonal_variation',
                typical_age_groups=['all_ages'],
                contagious=True,
                chronic=False
            ),
            
            'infectious_bacterial': DiseaseCategory(
                code='INF_BACTERIAL',
                name='Bacterial Infections',
                description='UTI, skin infections, bacterial pneumonia',
                keywords=['uti', 'urinary tract infection', 'skin infection', 'cellulitis',
                         'bacterial infection', 'abscess', 'wound infection', 'sepsis',
                         'bacterial pneumonia', 'strep throat'],
                icd10_codes=['N39', 'L03', 'A41', 'J15'],
                severity_indicators=['pus', 'severe', 'sepsis', 'high white count'],
                seasonal_pattern='no_pattern',
                typical_age_groups=['all_ages'],
                contagious=True,
                chronic=False
            ),
            
            'cardiovascular': DiseaseCategory(
                code='CARDIO',
                name='Cardiovascular Disorders',
                description='Hypertension, heart disease, chest pain',
                keywords=['hypertension', 'high blood pressure', 'chest pain', 'heart disease',
                         'cardiac', 'palpitations', 'heart attack', 'angina', 'arrhythmia',
                         'heart failure', 'coronary'],
                icd10_codes=['I10-I15', 'I20-I25', 'I30-I52'],
                severity_indicators=['severe chest pain', 'heart attack', 'acute'],
                seasonal_pattern='winter_increase',
                typical_age_groups=['adults', 'elderly'],
                contagious=False,
                chronic=True
            ),
            
            'endocrine_metabolic': DiseaseCategory(
                code='ENDO_META',
                name='Endocrine & Metabolic Disorders',
                description='Diabetes, thyroid disorders, metabolic syndrome',
                keywords=['diabetes', 'blood sugar', 'thyroid', 'hyperthyroid', 'hypothyroid',
                         'metabolic syndrome', 'obesity', 'weight gain', 'weight loss',
                         'fatigue', 'hormone', 'endocrine'],
                icd10_codes=['E10-E14', 'E00-E07', 'E88'],
                severity_indicators=['uncontrolled', 'severe', 'diabetic emergency'],
                seasonal_pattern='no_pattern',
                typical_age_groups=['adults', 'elderly'],
                contagious=False,
                chronic=True
            ),
            
            'musculoskeletal_acute': DiseaseCategory(
                code='MSK_ACUTE',
                name='Acute Musculoskeletal Injuries',
                description='Sprains, strains, acute back pain, sports injuries',
                keywords=['sprain', 'strain', 'acute back pain', 'muscle pain', 'joint pain',
                         'sports injury', 'trauma', 'fracture', 'dislocation', 'acute pain',
                         'injury', 'accident'],
                icd10_codes=['S00-S99', 'M25', 'M79'],
                severity_indicators=['severe pain', 'unable to move', 'fracture'],
                seasonal_pattern='summer_sports',
                typical_age_groups=['young_adults', 'adults'],
                contagious=False,
                chronic=False
            ),
            
            'musculoskeletal_chronic': DiseaseCategory(
                code='MSK_CHRONIC',
                name='Chronic Musculoskeletal Disorders',
                description='Arthritis, chronic back pain, fibromyalgia',
                keywords=['arthritis', 'chronic back pain', 'chronic pain', 'fibromyalgia',
                         'osteoarthritis', 'rheumatoid arthritis', 'joint stiffness',
                         'chronic joint pain', 'degenerative', 'chronic neck pain'],
                icd10_codes=['M15-M19', 'M05-M14', 'M79.3'],
                severity_indicators=['chronic', 'persistent', 'severe disability'],
                seasonal_pattern='winter_worse',
                typical_age_groups=['elderly', 'adults'],
                contagious=False,
                chronic=True
            ),
            
            'neurological': DiseaseCategory(
                code='NEURO',
                name='Neurological Disorders',
                description='Headaches, migraines, seizures, neuropathy',
                keywords=['headache', 'migraine', 'seizure', 'epilepsy', 'neuropathy',
                         'nerve pain', 'numbness', 'tingling', 'dizziness', 'vertigo',
                         'neurological', 'brain', 'stroke'],
                icd10_codes=['G43', 'G40', 'G60-G64', 'G93'],
                severity_indicators=['severe headache', 'status epilepticus', 'stroke'],
                seasonal_pattern='weather_sensitive',
                typical_age_groups=['all_ages'],
                contagious=False,
                chronic=True
            ),
            
            'dermatological': DiseaseCategory(
                code='DERM',
                name='Dermatological Conditions',
                description='Skin rashes, eczema, dermatitis, skin infections',
                keywords=['rash', 'eczema', 'dermatitis', 'skin infection', 'itching',
                         'skin condition', 'allergic reaction', 'hives', 'psoriasis',
                         'acne', 'skin lesion', 'wound'],
                icd10_codes=['L20-L30', 'L40-L45', 'L50-L54'],
                severity_indicators=['severe reaction', 'widespread', 'infected'],
                seasonal_pattern='seasonal_allergies',
                typical_age_groups=['all_ages'],
                contagious=False,
                chronic=False
            ),
            
            'mental_health': DiseaseCategory(
                code='MENTAL',
                name='Mental Health Disorders',
                description='Anxiety, depression, stress-related disorders',
                keywords=['anxiety', 'depression', 'stress', 'panic', 'mental health',
                         'psychological', 'mood disorder', 'insomnia', 'sleep disorder',
                         'emotional', 'psychiatric'],
                icd10_codes=['F32-F33', 'F40-F48', 'F51'],
                severity_indicators=['severe', 'suicidal', 'psychotic'],
                seasonal_pattern='winter_depression',
                typical_age_groups=['adults', 'young_adults'],
                contagious=False,
                chronic=True
            ),
            
            'pediatric_specific': DiseaseCategory(
                code='PEDIATRIC',
                name='Pediatric-Specific Conditions',
                description='Childhood diseases, developmental issues',
                keywords=['childhood disease', 'pediatric', 'developmental delay',
                         'growth issues', 'vaccination', 'infant', 'child development',
                         'teething', 'colic', 'diaper rash'],
                icd10_codes=['P00-P96', 'Q00-Q99'],
                severity_indicators=['failure to thrive', 'severe developmental delay'],
                seasonal_pattern='school_year',
                typical_age_groups=['infants', 'children'],
                contagious=False,
                chronic=False
            ),
            
            'reproductive_health': DiseaseCategory(
                code='REPRO',
                name='Reproductive Health',
                description='Gynecological, urological, reproductive issues',
                keywords=['gynecological', 'menstrual', 'pregnancy', 'reproductive',
                         'urological', 'prostate', 'sexual health', 'fertility',
                         'contraception', 'pelvic pain'],
                icd10_codes=['N70-N98', 'N40-N53', 'O00-O99'],
                severity_indicators=['severe bleeding', 'emergency', 'complications'],
                seasonal_pattern='no_pattern',
                typical_age_groups=['adults'],
                contagious=False,
                chronic=False
            )
        }
        
        return categories
    
    def _initialize_severity_keywords(self) -> Dict[str, List[str]]:
        """Initialize severity classification keywords"""
        
        return {
            'mild': ['mild', 'slight', 'minor', 'low grade', 'manageable', 'tolerable'],
            'moderate': ['moderate', 'significant', 'noticeable', 'concerning', 'persistent'],
            'severe': ['severe', 'intense', 'excruciating', 'unbearable', 'emergency', 
                      'acute', 'critical', 'hospitalization', 'admission'],
            'chronic': ['chronic', 'long-term', 'persistent', 'ongoing', 'recurrent',
                       'long-standing', 'years', 'months']
        }
    
    def classify_disease(self, chief_complaint: str, management: str = "", 
                        patient_age: Optional[int] = None) -> Dict[str, any]:
        """
        Classify disease based on chief complaint and management
        
        Returns:
            Dict containing category, confidence, severity, and additional metadata
        """
        
        if not chief_complaint:
            return self._default_classification()
        
        # Normalize text
        text = f"{chief_complaint} {management}".lower().strip()
        
        # Calculate scores for each category
        category_scores = {}
        for cat_code, category in self.disease_categories.items():
            score = self._calculate_category_score(text, category, patient_age)
            if score > 0:
                category_scores[cat_code] = score
        
        if not category_scores:
            return self._default_classification()
        
        # Get best match
        best_category_code = max(category_scores, key=category_scores.get)
        best_category = self.disease_categories[best_category_code]
        confidence = min(category_scores[best_category_code] / 10.0, 1.0)  # Normalize to 0-1
        
        # Determine severity
        severity = self._determine_severity(text)
        
        # Additional metadata
        metadata = {
            'seasonal_pattern': best_category.seasonal_pattern,
            'contagious': best_category.contagious,
            'chronic': best_category.chronic,
            'typical_age_groups': best_category.typical_age_groups,
            'icd10_codes': best_category.icd10_codes
        }
        
        return {
            'category_code': best_category_code,
            'category_name': best_category.name,
            'description': best_category.description,
            'confidence': confidence,
            'severity': severity,
            'metadata': metadata,
            'all_scores': category_scores
        }
    
    def _calculate_category_score(self, text: str, category: DiseaseCategory, 
                                 patient_age: Optional[int] = None) -> float:
        """Calculate matching score for a category"""
        
        score = 0.0
        
        # Keyword matching with weighted scoring
        for keyword in category.keywords:
            if keyword in text:
                # Longer keywords get higher scores
                weight = len(keyword.split()) * 2
                score += weight
        
        # Age group bonus
        if patient_age and self._age_matches_category(patient_age, category.typical_age_groups):
            score += 1.0
        
        # Severity indicator bonus
        for indicator in category.severity_indicators:
            if indicator in text:
                score += 1.5
        
        return score
    
    def _age_matches_category(self, age: int, age_groups: List[str]) -> bool:
        """Check if patient age matches category's typical age groups"""
        
        age_mapping = {
            'infants': (0, 2),
            'children': (2, 12),
            'young_adults': (13, 30),
            'adults': (18, 65),
            'elderly': (65, 120),
            'all_ages': (0, 120),
            'immunocompromised': (0, 120)  # Special category
        }
        
        for group in age_groups:
            if group in age_mapping:
                min_age, max_age = age_mapping[group]
                if min_age <= age <= max_age:
                    return True
        
        return False
    
    def _determine_severity(self, text: str) -> str:
        """Determine severity level from text"""
        
        severity_scores = {}
        
        for severity, keywords in self.severity_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                severity_scores[severity] = score
        
        if not severity_scores:
            return 'mild'
        
        # Return highest scoring severity
        return max(severity_scores, key=severity_scores.get)
    
    def _default_classification(self) -> Dict[str, any]:
        """Return default classification for unmatched cases"""
        
        return {
            'category_code': 'UNCLASSIFIED',
            'category_name': 'Unclassified',
            'description': 'Unable to classify based on available information',
            'confidence': 0.0,
            'severity': 'mild',
            'metadata': {
                'seasonal_pattern': 'no_pattern',
                'contagious': False,
                'chronic': False,
                'typical_age_groups': ['all_ages'],
                'icd10_codes': []
            },
            'all_scores': {}
        }
    
    def get_category_statistics(self, classifications: List[Dict]) -> Dict[str, any]:
        """Generate statistics from a list of classifications"""
        
        if not classifications:
            return {}
        
        # Count by category
        category_counts = {}
        severity_counts = {}
        seasonal_patterns = {}
        
        for classification in classifications:
            # Category counts
            cat_code = classification.get('category_code', 'UNCLASSIFIED')
            category_counts[cat_code] = category_counts.get(cat_code, 0) + 1
            
            # Severity counts
            severity = classification.get('severity', 'mild')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Seasonal patterns
            pattern = classification.get('metadata', {}).get('seasonal_pattern', 'no_pattern')
            seasonal_patterns[pattern] = seasonal_patterns.get(pattern, 0) + 1
        
        return {
            'total_cases': len(classifications),
            'category_distribution': category_counts,
            'severity_distribution': severity_counts,
            'seasonal_patterns': seasonal_patterns,
            'average_confidence': sum(c.get('confidence', 0) for c in classifications) / len(classifications)
        }
    
    def get_all_categories(self) -> Dict[str, Dict]:
        """Get all available disease categories with metadata"""
        
        return {
            code: {
                'name': cat.name,
                'description': cat.description,
                'seasonal_pattern': cat.seasonal_pattern,
                'contagious': cat.contagious,
                'chronic': cat.chronic,
                'typical_age_groups': cat.typical_age_groups,
                'keyword_count': len(cat.keywords)
            }
            for code, cat in self.disease_categories.items()
        }

# Global instance
disease_classifier = EnhancedDiseaseClassifier()
