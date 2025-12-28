# created with claude 

import re
import json
import os
from typing import List, Dict, Optional

obj = [
    "AktG",
    "ABGB",
    "GBG",
    "ASVG",
    "AVG",
    "AngG",
    "ArbVG",
    "AZG",
    "BSVG",
    "BBG",
    "B-VG",
    "DSG",
    "E-GovG",
    "EStG",
    "EpiG",
    "Europäische-Menschenrechtskonvention",
    "FSG",
    "GewO",
    "GSVG",
    "GmbHG",
    "KSchG",
    "KFG",
    "MeldeG",
    "MRG",
    "NRWO",
    "SchUG",
    "SPG",
    "StbG",
    "StPO",
    "StVO",
    "TSchG",
    "UStG",
    "UG",
    "UGB",
    "VerG",
    "VStG",
    "WEG",
    "ZPO",
    "ZustG"
]



class AustrianLawReferenceFinder:
    
    def parse_reference(self, ref_text: str) -> Optional[Dict]:
        """
        Parse a legal reference string into components.
        
        Handles formats like:
        - §366 Abs1 Z3 GewO 1994
        - §16 Abs2 und Abs5 Niederlassungs- und AufenthaltsG (NAG)
        - §1 der Integrationsvereinungs-Verordnung [IV-VO], BGBl II 449/2005
        - Art144 B-VG
        - Art20 Abs1 und Art77 B-VG
        - Art20 Abs2 Z1 B-VG
        - §20
        - §21 VStG
        - § 1202
        
        Returns:
            Dict with parsed components or None if parsing fails
        """
        # Pattern for Article references (Art144 B-VG, Art20 Abs1 B-VG, etc.)
        art_pattern = r'Art(?:ikel)?\s*(\d+[a-z]?)\s*(?:Abs\.?\s*(\d+))?\s*(?:Z\.?\s*(\d+))?\s*([\w\-]+)?'
        match = re.search(art_pattern, ref_text, re.IGNORECASE)
        if match:
            return {
                'type': 'article',
                'number': match.group(1),
                'absatz': match.group(2),
                'ziffer': match.group(3),
                'law': match.group(4).strip() if match.group(4) else None,
                'original': ref_text
            }
        
        # Pattern for § references with complex law names
        # §16 Abs2 und Abs5 Niederlassungs- und AufenthaltsG (NAG)
        para_complex = r'§\s*(\d+[a-z]?)\s*(?:Abs\.?\s*(\d+))?\s*(?:(?:und|oder)\s*Abs\.?\s*(\d+))?\s*(?:Z\.?\s*(\d+))?\s*(.+?)(?:,\s*BGBl)?'
        match = re.search(para_complex, ref_text, re.IGNORECASE)
        if match:
            absatz_list = [match.group(2)] if match.group(2) else []
            if match.group(3):
                absatz_list.append(match.group(3))
            
            return {
                'type': 'paragraph',
                'number': match.group(1),
                'absatz': absatz_list if absatz_list else None,
                'ziffer': match.group(4),
                'law': match.group(5).strip() if match.group(5) else None,
                'original': ref_text
            }
        
        # Standard § pattern: §366 Abs1 Z3 GewO 1994
        para_standard = r'§\s*(\d+[a-z]?)\s*(?:Abs\.?\s*(\d+))?\s*(?:Z\.?\s*(\d+))?\s*([A-Za-zÄÖÜäöüß]+(?:\s+\d{4})?)?'
        match = re.search(para_standard, ref_text, re.IGNORECASE)
        if match:
            return {
                'type': 'paragraph',
                'number': match.group(1),
                'absatz': [match.group(2)] if match.group(2) else None,
                'ziffer': match.group(3),
                'law': match.group(4).strip() if match.group(4) else None,
                'original': ref_text
            }
        
        # Simpler pattern for just §20
        para_simple = r'§\s*(\d+[a-z]?)'
        match = re.search(para_simple, ref_text, re.IGNORECASE)
        if match:
            return {
                'type': 'paragraph',
                'number': match.group(1),
                'absatz': None,
                'ziffer': None,
                'law': None,
                'original': ref_text
            }
        
        return None
    
    def find_in_text(self, parsed: Dict, context_lines: int = 15) -> List[Dict]:
        """
        Find the parsed reference in the law text.
        
        Args:
            parsed: Parsed reference dictionary
            context_lines: Number of lines to include after the match
            
        Returns:
            List of matches with line number and context
        """
        if not parsed:
            return []
        
        results = []
        number = parsed['number']
        ref_type = parsed.get('type', 'paragraph')
        
        # Create search patterns based on type
        if ref_type == 'article':
            patterns = [
                re.compile(rf'^Art(?:ikel)?\s*{re.escape(number)}\.?\s*$', re.IGNORECASE),
                re.compile(rf'^Art(?:ikel)?\s*{re.escape(number)}\.\s+\w+', re.IGNORECASE),
                re.compile(rf'Art(?:ikel)?\s*{re.escape(number)}\b', re.IGNORECASE),
            ]
        else:  # paragraph
            patterns = [
                re.compile(rf'^§\s*{re.escape(number)}\.?\s*$', re.IGNORECASE),
                re.compile(rf'^§\s*{re.escape(number)}\.\s+\w+', re.IGNORECASE),
                re.compile(rf'§\s*{re.escape(number)}\b', re.IGNORECASE),
            ]
        
        # If we have Absatz, add more specific patterns
        if parsed['absatz']:
            if isinstance(parsed['absatz'], list):
                # Multiple Absatz values
                for abs_val in parsed['absatz']:
                    if abs_val:
                        if ref_type == 'article':
                            patterns.insert(0, 
                                re.compile(rf'Art(?:ikel)?\s*{re.escape(number)}\s*Abs\.?\s*{abs_val}', re.IGNORECASE)
                            )
                        else:
                            patterns.insert(0, 
                                re.compile(rf'§\s*{re.escape(number)}\s*Abs\.?\s*{abs_val}', re.IGNORECASE)
                            )
            else:
                # Single Absatz value
                if ref_type == 'article':
                    patterns.insert(0, 
                        re.compile(rf'Art(?:ikel)?\s*{re.escape(number)}\s*Abs\.?\s*{parsed["absatz"]}', re.IGNORECASE)
                    )
                else:
                    patterns.insert(0, 
                        re.compile(rf'§\s*{re.escape(number)}\s*Abs\.?\s*{parsed["absatz"]}', re.IGNORECASE)
                    )
        
        for i, line in enumerate(self.lines):
            matched = False
            for pattern in patterns:
                if pattern.search(line):
                    matched = True
                    break
            
            if matched:
                # Found a match - collect context
                start = max(0, i)
                end = min(len(self.lines), i + context_lines)
                context = '\n'.join(self.lines[start:end])
                
                results.append({
                    'line_number': i + 1,
                    'context': context,
                    'matched_line': line.strip()
                })
        
        return results
    
    def search_references(self, references: List[Dict]) -> List[Dict]:
        """
        Search for all references in the law text.
        
        Args:
            references: List of reference dicts with 'text' key
            
        Returns:
            List of results with parsed info and matches
        """
        results = []
        
        for ref in references:
            print(ref)
            filename = ""
            for gestez in obj:
                if gestez in ref["text"]:
                    filename = gestez
                    break

            if filename == "":
                continue
            with open(f"./law_txt/{filename}.txt", 'r', encoding='utf-8') as f:
                law_text = f.read()
                self.law_text = law_text
                self.lines = law_text.split('\n')

            ref_text = ref.get('text', '')
            parsed = self.parse_reference(ref_text)
            
            if parsed:
                matches = self.find_in_text(parsed)
                results.append({
                    'reference': ref_text,
                    'parsed': parsed,
                    'matches': matches,
                    'status': 'found' if matches else 'not_found'
                })
            else:
                results.append({
                    'reference': ref_text,
                    'parsed': None,
                    'matches': [],
                    'status': 'parse_error'
                })
        
        return results
    
    def format_parsed(self, parsed: Dict) -> str:
        """Format parsed reference for display"""
        if not parsed:
            return "Unable to parse"
        
        result = []
        if parsed['type'] == 'article':
            result.append(f"Art{parsed['number']}")
        else:
            result.append(f"§{parsed['number']}")
        
        if parsed['absatz']:
            if isinstance(parsed['absatz'], list):
                abs_str = ' und Abs'.join(parsed['absatz'])
                result.append(f"Abs{abs_str}")
            else:
                result.append(f"Abs{parsed['absatz']}")
        
        if parsed['ziffer']:
            result.append(f"Z{parsed['ziffer']}")
        
        if parsed['law']:
            result.append(parsed['law'])
        
        return ' '.join(result)


# Example usage
if __name__ == '__main__':

    directory = "data_extracted"

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        with open(f"./data_extracted/{filename}", 'r', encoding='utf-8') as f:
            data = json.load(f)
            references = data['references']

        # Create finder and search
        finder = AustrianLawReferenceFinder()
        results = finder.search_references(references)

        # Print results
        for result in results:
            print(f"\n{'='*80}")
            print(f"Reference: {result['reference']}")
            print(f"Status: {result['status']}")

            if result['parsed']:
                print(f"Parsed: {finder.format_parsed(result['parsed'])}")

            if result['matches']:
                for match in result['matches']:
                    print(f"\n--- Found at line {match['line_number']} ---")
                    print(f"Matched: {match['matched_line']}")
                    print(f"\nContext:")
                    print(match['context'][:500])
                    if len(match['context']) > 500:
                        print("...")
            else:
                if result['status'] == 'not_found':
                    print("Not found in text")

        # Save results to JSON
        with open(f"./matchings/{filename}", 'w+', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)