#!/usr/bin/env python3
"""
Climate Research Variable Filter
Specialized for extracting measurable variables from climate science papers
Optimized for Arctic, polar, and climate change research
"""

import re
import json
import ollama
from typing import List, Dict, Set

class VariableFilter:
    def __init__(self):
        """
        Initialize climate-specific variable filter
        Comprehensive coverage of climate research terminology
        """
        
        # Core climate measurements and variables
        self.variable_indicators = {
            # Atmospheric variables
            "atmospheric": [
                "temperature", "air temperature", "surface temperature", "atmospheric temperature",
                "pressure", "atmospheric pressure", "sea level pressure", "barometric pressure",
                "humidity", "relative humidity", "specific humidity", "vapor pressure",
                "wind speed", "wind direction", "wind stress", "gust", "circulation",
                "precipitation", "rainfall", "snowfall", "rain rate", "snow accumulation",
                "cloud cover", "cloud fraction", "cloud height", "cloud type",
                "radiation", "shortwave radiation", "longwave radiation", "net radiation",
                "solar radiation", "infrared radiation", "uv radiation",
                "visibility", "atmospheric opacity", "aerosol optical depth"
            ],
            
            # Cryosphere (ice/snow) variables
            "cryosphere": [
                "sea ice extent", "sea ice concentration", "sea ice thickness", "sea ice volume",
                "sea ice age", "ice velocity", "ice drift", "ice deformation",
                "snow depth", "snow water equivalent", "snow density", "snow cover",
                "glacier mass", "glacier velocity", "glacier retreat", "ice sheet elevation",
                "permafrost temperature", "active layer thickness", "thaw depth",
                "ice core", "firn density", "melt rate", "freeze-up", "break-up",
                "fast ice", "pack ice", "marginal ice zone", "polynya", "lead fraction",
                "ice albedo", "snow albedo", "surface melt", "basal melt"
            ],
            
            # Ocean/Marine variables
            "oceanic": [
                "sea surface temperature", "sst", "ocean temperature", "water temperature",
                "sea level", "sea level rise", "sea surface height", "storm surge",
                "salinity", "sea surface salinity", "ocean salinity", "freshwater flux",
                "ocean current", "current velocity", "current direction", "circulation pattern",
                "mixed layer depth", "thermocline depth", "halocline", "pycnocline",
                "wave height", "significant wave height", "wave period", "wave direction",
                "ocean heat content", "heat flux", "latent heat", "sensible heat",
                "upwelling", "downwelling", "vertical velocity", "mixing rate",
                "dissolved oxygen", "ocean acidification", "ph", "alkalinity",
                "chlorophyll", "primary production", "phytoplankton", "biomass"
            ],
            
            # Carbon cycle and greenhouse gases
            "carbon_cycle": [
                "co2", "carbon dioxide", "co2 concentration", "atmospheric co2",
                "methane", "ch4", "methane concentration", "methane flux",
                "carbon flux", "carbon storage", "carbon sequestration", "carbon sink",
                "greenhouse gas", "ghg", "emission rate", "emission factor",
                "carbon isotope", "δ13c", "δ14c", "radiocarbon",
                "dissolved organic carbon", "doc", "particulate organic carbon", "poc",
                "soil carbon", "vegetation carbon", "carbon stock", "carbon budget",
                "net primary production", "npp", "gross primary production", "gpp",
                "respiration rate", "ecosystem respiration", "soil respiration"
            ],
            
            # Climate indices and patterns
            "climate_indices": [
                "nao", "north atlantic oscillation", "ao", "arctic oscillation",
                "enso", "el niño", "la niña", "southern oscillation index",
                "pdo", "pacific decadal oscillation", "amo", "atlantic multidecadal oscillation",
                "sam", "southern annular mode", "iod", "indian ocean dipole",
                "warming rate", "cooling rate", "temperature trend", "temperature anomaly",
                "heat wave", "cold wave", "extreme event frequency", "return period",
                "climate sensitivity", "equilibrium climate sensitivity", "transient climate response",
                "radiative forcing", "feedback factor", "climate feedback"
            ],
            
            # Energy balance and fluxes
            "energy_balance": [
                "energy flux", "heat flux", "radiative flux", "turbulent flux",
                "latent heat flux", "sensible heat flux", "ground heat flux",
                "net radiation", "radiation balance", "energy balance",
                "albedo", "surface albedo", "planetary albedo", "reflectance",
                "emissivity", "transmittance", "absorptance", "optical depth",
                "downward radiation", "upward radiation", "outgoing longwave radiation",
                "surface energy budget", "atmospheric energy transport", "oceanic heat transport"
            ]
        }
        
        # Climate-specific exclusions (minimal for climate papers)
        self.exclude_terms = [
            # Only exclude pure organizational/document terms
            "university", "institute", "department", "laboratory",
            "introduction", "conclusion", "abstract", "reference", "appendix",
            "table", "figure", "equation", "section", "chapter",
            "author", "publication", "journal", "conference",
            
            # Pure methodology terms that aren't measurements
            "hypothesis", "theory", "review", "survey", "questionnaire"
        ]
        
        # Climate research concepts that ARE variables (always keep)
        self.climate_concepts = [
            # Climate change indicators
            "climate change", "global warming", "global cooling", "climate variability",
            "climate trend", "climate shift", "regime shift", "tipping point",
            "tipping element", "climate extreme", "compound event",
            
            # Arctic-specific terms
            "arctic amplification", "polar amplification", "arctic warming",
            "arctic sea ice", "arctic ocean", "arctic front", "arctic haze",
            "polar vortex", "polar night", "polar day", "midnight sun",
            
            # Ice-related processes
            "ice-albedo feedback", "ice sheet dynamics", "ice shelf collapse",
            "calving rate", "grounding line", "ice stream", "ice tongue",
            "sea ice", "land ice", "multi-year ice", "first-year ice",
            
            # Climate models and projections
            "climate model", "earth system model", "gcm", "rcm",
            "climate projection", "climate scenario", "rcp", "ssp",
            "cmip5", "cmip6", "ensemble mean", "model bias",
            
            # Feedback mechanisms
            "positive feedback", "negative feedback", "climate feedback",
            "water vapor feedback", "cloud feedback", "lapse rate feedback",
            "planck feedback", "carbon cycle feedback",
            
            # Paleoclimate indicators
            "proxy data", "ice core record", "tree ring", "coral record",
            "sediment core", "pollen record", "isotope ratio",
            
            # Impact variables
            "sea level rise", "coastal erosion", "storm intensity",
            "drought frequency", "flood risk", "wildfire activity",
            "ecosystem shift", "species migration", "phenology",
            
            # Mitigation/Adaptation
            "carbon capture", "renewable energy", "mitigation potential",
            "adaptation capacity", "climate resilience", "vulnerability index"
        ]
        
        # Units commonly used in climate science (strong variable indicators)
        self.climate_units = [
            "°c", "k", "kelvin", "celsius", "fahrenheit",
            "ppm", "ppb", "ppmv", "ppbv",  # Gas concentrations
            "w/m2", "w/m²", "wm-2",  # Radiation/flux
            "mm/day", "mm/yr", "cm/yr", "m/yr",  # Precipitation/melt rates
            "psu", "‰",  # Salinity
            "mb", "hpa", "kpa",  # Pressure
            "m/s", "km/h", "knots",  # Wind/current speed
            "km2", "km²", "106 km2",  # Ice extent
            "gt", "pg", "tg",  # Mass (gigatonne, petagram, teragram)
            "sv", "mw", "tw"  # Ocean/energy transport
        ]
    
    def is_variable(self, keyword: str) -> bool:
        """
        Determine if a keyword is a climate-related measurable variable
        Optimized for climate science research papers
        """
        kw_lower = keyword.lower().strip()
        
        # Priority 1: Check if it's a known climate concept (ALWAYS KEEP)
        for concept in self.climate_concepts:
            if concept in kw_lower or kw_lower in concept:
                return True
        
        # Priority 2: Check against comprehensive variable indicators
        for category, terms in self.variable_indicators.items():
            for term in terms:
                # Exact match or term is part of keyword
                if term == kw_lower or term in kw_lower:
                    return True
                # Keyword is part of term (catches abbreviations)
                if len(kw_lower) > 2 and kw_lower in term:
                    return True
        
        # Priority 3: Check for climate units (strong indicator)
        for unit in self.climate_units:
            if unit in kw_lower.replace(" ", ""):
                return True
        
        # Check for numerical patterns with units
        if re.search(r'\d+[\s-]?(°[CF]|K|m/s|km/h|mm|cm|m|kg|g|%|ppm|ppb|w/m)', kw_lower):
            return True
        
        # Check for parenthetical units
        if re.search(r'\([^)]*(?:°[CF]|K|ppm|ppb|m/s|w/m|%)[^)]*\)', keyword):
            return True
        
        # Priority 4: Climate-specific keyword patterns
        climate_patterns = [
            r'^\w+\s+(?:temperature|warming|cooling|change)$',
            r'^\w+\s+(?:ice|snow|glacier|permafrost)$',
            r'^\w+\s+(?:flux|flow|transport|exchange)$',
            r'^\w+\s+(?:concentration|content|storage)$',
            r'^\w+\s+(?:anomaly|trend|variability|oscillation)$',
            r'^(?:annual|seasonal|monthly|daily)\s+\w+$',
            r'^\w+\s+(?:maximum|minimum|mean|average)$'
        ]
        
        for pattern in climate_patterns:
            if re.match(pattern, kw_lower):
                return True
        
        # Priority 5: Compound climate terms
        compound_indicators = [
            "_temperature", "_pressure", "_velocity", "_speed", "_rate",
            "_flux", "_content", "_concentration", "_depth", "_height",
            "_thickness", "_extent", "_area", "_volume", "_mass",
            "_anomaly", "_trend", "_index", "_ratio", "_fraction"
        ]
        
        for indicator in compound_indicators:
            if indicator in kw_lower.replace(" ", "_"):
                return True
        
        # Priority 6: Check exclusions (strict match only)
        if kw_lower in self.exclude_terms:
            return False
        
        # Priority 7: Climate domain heuristics
        # If it contains key climate words, likely a variable
        climate_keywords = [
            "temperature", "pressure", "humidity", "precipitation",
            "ice", "snow", "glacier", "permafrost", "frost",
            "ocean", "sea", "marine", "coastal", "arctic", "antarctic", "polar",
            "atmosphere", "atmospheric", "stratosphere", "troposphere",
            "carbon", "co2", "methane", "greenhouse", "emission",
            "radiation", "albedo", "flux", "heat", "energy",
            "wind", "storm", "cyclone", "hurricane", "typhoon",
            "drought", "flood", "extreme", "hazard",
            "climate", "weather", "meteorological",
            "warming", "cooling", "melting", "freezing",
            "rise", "decline", "increase", "decrease", "change"
        ]
        
        for climate_word in climate_keywords:
            if climate_word in kw_lower:
                return True
        
        # Priority 8: Statistical/measurement terms with potential climate context
        if len(kw_lower.split()) >= 2:
            measurement_terms = [
                "mean", "average", "median", "maximum", "minimum",
                "anomaly", "deviation", "variance", "trend", "gradient",
                "rate", "ratio", "index", "factor", "coefficient",
                "positive", "negative", "increasing", "decreasing"
            ]
            for term in measurement_terms:
                if term in kw_lower:
                    return True
        
        # Priority 9: Common climate abbreviations and acronyms
        climate_acronyms = [
            "sst", "sat", "slp", "rh", "aod", "toa", "ndvi",
            "amoc", "itcz", "mjo", "qbo", "nam", "sam",
            "ghg", "gwp", "eei", "erf", "tcr", "ecs"
        ]
        if kw_lower in climate_acronyms:
            return True
        
        # Default: For climate papers, be inclusive
        # Single words > 3 chars that aren't in exclusions are likely variables
        if len(kw_lower.split()) == 1 and len(kw_lower) > 3:
            if not any(x in kw_lower for x in ["the", "and", "or", "but", "for", "with", "from", "into"]):
                return True
                
        return False
    
    def filter_variables(self, keywords: List[str]) -> Dict[str, List[str]]:
        """
        Filter keywords to separate climate variables from non-variables
        Optimized for climate science papers
        """
        variables = []
        non_variables = []
        
        for kw in keywords:
            if self.is_variable(kw):
                variables.append(kw)
            else:
                non_variables.append(kw)
        
        return {
            "variables": variables,
            "non_variables": non_variables,
            "variable_rate": len(variables) / len(keywords) * 100 if keywords else 0
        }
    
    def categorize_climate_variables(self, variables: List[str]) -> Dict[str, List[str]]:
        """
        Categorize climate variables by type for better understanding
        Useful for research paper analysis
        """
        categorized = {
            "atmospheric": [],
            "cryosphere": [],
            "oceanic": [],
            "carbon_cycle": [],
            "climate_indices": [],
            "energy_balance": [],
            "other": []
        }
        
        for var in variables:
            var_lower = var.lower()
            found_category = False
            
            # Check each category
            for category, terms in self.variable_indicators.items():
                for term in terms:
                    if term in var_lower:
                        categorized[category].append(var)
                        found_category = True
                        break
                if found_category:
                    break
            
            # If no category found, add to other
            if not found_category:
                categorized["other"].append(var)
        
        # Remove empty categories
        return {k: v for k, v in categorized.items() if v}
    
    def enhance_with_llm(self, keywords: List[str], text_context: str = "") -> List[str]:
        """
        Use LLM to better identify variables/features
        """
        prompt = f"""
From the following keywords extracted from a climate/environmental research paper,
identify ONLY the measurable variables and features (parameters that can be measured or quantified).

Keywords: {', '.join(keywords)}

Rules:
1. Include: temperature, pressure, speed, concentration, thickness, density, etc.
2. Include: ice extent, snow coverage, precipitation rate, humidity levels, etc.
3. Exclude: locations (Arctic, Pacific), organizations (NASA, NOAA), methods (satellite, model)
4. Exclude: general concepts (study, research, climate change as a concept)

Return ONLY measurable variables/features as a simple list, one per line.
"""
        
        try:
            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response['message']['content']
            variables = []
            
            for line in content.strip().split('\n'):
                line = line.strip('- •*').strip()
                if line and len(line) > 2:
                    variables.append(line.lower())
            
            return variables
            
        except Exception as e:
            print(f"LLM enhancement error: {e}")
            return []
    
    def get_filtered_graph(self, neo4j_nodes: List[str], neo4j_edges: List[Dict]) -> Dict:
        """
        Filter the knowledge graph to keep only variable nodes and their relationships
        """
        # Filter nodes to get variables
        result = self.filter_variables(list(neo4j_nodes))
        variables = set([v.lower().strip() for v in result["variables"]])
        
        print(f"\n📊 Variable Filtering Results:")
        print(f"   Original nodes: {len(neo4j_nodes)}")
        print(f"   Variables identified: {len(variables)}")
        print(f"   Filtered out: {len(result['non_variables'])}")
        
        # Filter edges to keep only those between variables
        filtered_edges = []
        for edge in neo4j_edges:
            source = edge.get("source", "").lower().strip()
            target = edge.get("target", "").lower().strip()
            
            if source in variables and target in variables:
                filtered_edges.append(edge)
        
        print(f"   Original edges: {len(neo4j_edges)}")
        print(f"   Filtered edges: {len(filtered_edges)}")
        
        return {
            "variables": list(variables),
            "relationships": filtered_edges,
            "filtered_out": result["non_variables"]
        }


def extract_variables_only(neo4j_nodes, neo4j_edges, enhance_with_llm=False, text_context=""):
    """
    Simple function to extract only variables from knowledge graph
    """
    filter = VariableFilter()
    
    # Basic filtering
    filtered_graph = filter.get_filtered_graph(neo4j_nodes, neo4j_edges)
    
    # Optional: Enhance with LLM
    if enhance_with_llm and text_context:
        llm_variables = filter.enhance_with_llm(list(neo4j_nodes), text_context[:2000])
        
        # Combine results
        all_variables = set(filtered_graph["variables"] + llm_variables)
        filtered_graph["variables"] = list(all_variables)
        
        print(f"\n   Enhanced with LLM: {len(all_variables)} total variables")
    
    return filtered_graph


if __name__ == "__main__":
    # Example usage
    sample_nodes = [
        "temperature", "NASA", "ice thickness", "Arctic Ocean", "wind speed",
        "satellite observation", "pressure gradient", "NOAA", "salinity",
        "methodology", "heat flux", "climate model", "precipitation rate",
        "research study", "ocean current", "albedo", "data analysis"
    ]
    
    sample_edges = [
        {"source": "temperature", "relation": "AFFECTS", "target": "ice thickness"},
        {"source": "NASA", "relation": "USES", "target": "satellite observation"},
        {"source": "wind speed", "relation": "INFLUENCES", "target": "ocean current"},
        {"source": "temperature", "relation": "CORRELATES", "target": "pressure gradient"}
    ]
    
    # Test the filter
    filter = VariableFilter()
    result = filter.get_filtered_graph(sample_nodes, sample_edges)
    
    print("\n✅ Variables extracted:")
    for var in sorted(result["variables"]):
        print(f"   - {var}")
    
    print("\n❌ Filtered out (non-variables):")
    for item in sorted(result["filtered_out"])[:10]:
        print(f"   - {item}")