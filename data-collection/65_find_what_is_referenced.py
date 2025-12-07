import json


obj = {
    "Aktiengesetz": "AktG",
    "Allgemeines bürgerliches Gesetzbuch": "ABGB",
    "Allgemeines Grundbuchsgesetz 1955": "GBG",
    "Allgemeines Sozialversicherungsgesetz": "ASVG",
    "Allgemeines Verwaltungsverfahrensgesetz 1991": "AVG",
    "Angestelltengesetz": "AngG",
    "Arbeitsverfassungsgesetz": "ArbVG",
    "Arbeitszeitgesetz": "AZG",
    "Bauern-Sozialversicherungsgesetz": "BSVG",
    "Bundesbehindertengesetz": "BBG",
    "Bundes-Verfassungsgesetz": "B-VG",
    "Datenschutzgesetz": "DSG",
    "E-Government-Gesetz" : "E-GovG",
    "Einkommensteuergesetz 1988": "EStG",
    "Epidemiegesetz 1950": "EpiG",
    "Europäische Menschenrechtskonvention": "Europäische Menschenrechtskonvention",
    "Führerscheingesetz": "FSG",
    "Gewerbeordnung 1994": "GewO 1994",
    "Gewerbliches Sozialversicherungsgesetz": "GSVG",
    "GmbH-Gesetz" : "GmbHG",
    "Konsumentenschutzgesetz": "KSchG",
    "Kraftfahrgesetz 1967" : "KFG 1967",
    "Meldegesetz 1991": "MeldeG",
    "Mietrechtsgesetz": "MRG",
    "Nationalrats-Wahlordnung":  "NRWO",
    "Schulunterrichtsgesetz" : "SchUG",
    "Sicherheitspolizeigesetz": "SPG",
    "Staatsbürgerschaftsgesetz 1985":  "StbG",
    "Strafgesetzbuch":"StGB",
    "Strafprozeßordnung 1975": "StPO",
    "Straßenverkehrsordnung 1960" : "StVO",
    "Tierschutzgesetz" : "TSchG",
    "Umsatzsteuergesetz 1994": "UStG",
    "Universitätsgesetz 2002": "UG",
    "Unternehmensgesetzbuch": "UGB",
    "Vereinsgesetz 2002":  "VerG",
    "Verwaltungsstrafgesetz 1991": "VStG",
    "Wohnungseigentumsgesetz 2002":  "WEG",
    "Zivilprozessordnung": "ZPO",
    "Zustellgesetz": "ZustG"
}

with open("./filtered_referneces.json", "r") as f:
    list_of_filtered = json.load(f)


results = {}

for (k,v) in obj.items():
    results[v] = 0

for item in list_of_filtered:
    with open("./data_extracted/"+item, "r") as f:
        object_in_file = json.load(f)
        for a in object_in_file["references"]:
            for (k,v) in obj.items():
                if v in a["text"]:
                    results[v] += 1

print(json.dumps(results, indent=2))