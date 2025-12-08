SCHEMA_BIODIV = """

Biodiversity 	HAS PROPERTY	Species
                HAS PROPERTY	Ecosystem
                HAS PROPERTY	Gene

Species	HAS PROPERTY	Characteristics	HAS PROPERTY	Habitat
                                        HAS PROPERTY	Feeding regime
                                        HAS PROPERTY	Life history

        HAS PROPERTY	Diversity			HAS PROPERTY	status
                                            HAS PROPERTY	trend
        HAS PROPERTY	Population	        HAS PROPERTY	Density	HAS PROPERTY	status
                                                                    HAS PROPERTY	trend
                                            HAS PROPERTY	Distribution	HAS PROPERTY	status
                                                                            HAS PROPERTY	trend
                                            HAS PROPERTY	Size	        HAS PROPERTY	status
                                                                            HAS PROPERTY	trend
        HAS PROPERTY	Distribution	    HAS PROPERTY	Space	        HAS PROPERTY	status
                                                                            HAS PROPERTY	trend
                                            HAS PROPERTY	Elevation	    HAS PROPERTY	status
                                                                            HAS PROPERTY	trend
                                            HAS PROPERTY    Time	        HAS PROPERTY	trend
        HAS PROPERTY	Conservation Status			                        HAS PROPERTY	status
                                                                            HAS PROPERTY	trend

        IS AFFECTED BY	Driver

        IS DETERMINING 	Functions
                        Ecosystem services


Drivers	HAS TYPE	Climate			                        HAS PROPERTY	trend
                                    HAS TYPE	Temperature	HAS PROPERTY	status
                                                            HAS EFFECT		HAS PROPERTY	trend
        HAS TYPE	Precipitation	HAS PROPERTY	status
                                    HAS PROPERTY	trend
                                    HAS EFFECT		HAS PROPERTY	trend
        HAS TYPE	Wind	        HAS PROPERTY	status
                                    HAS PROPERTY	trend
                                    HAS EFFECT		HAS PROPERTY	trend
        HAS TYPE	Drought	        HAS PROPERTY	status
                                    HAS PROPERTY	trend
                                    HAS EFFECT		HAS PROPERTY	trend
        HAS TYPE	Extreme events	HAS PROPERTY	status
                                    HAS PROPERTY	trend
                                    HAS EFFECT		HAS PROPERTY	trend
"""


SCHEMA_TYPES = ["BIODIVERSITY", "SPECIES", "GENE", "ECOSYSTEM", "HABITAT", "FEEDING REGIME", "LIFE HISTORY", "DIVERSITY",
                "POPULATION DENSITY trend", "POPULATION DENSITY status", "POPULATION DISTRIBUTION trend",
                "POPULATION DISTRIBUTION status", "POPULATION SIZE trend", "POPULATION SIZE status",
                "POPULATION DISTRIBUTION SPACE trend", "POPULATION DISTRIBUTION SPACE status",
                "POPULATION DISTRIBUTION ELEVATION trend", "POPULATION DISTRIBUTION ELEVATION status",
                "POPULATION DISTRIBUTION TIME trend", "POPULATION DISTRIBUTION TIME status",
                "CONSERVATION STATUS trend", "CONSERVATION STATUS status", "DRIVER", "FUNCTIONS",
                "ECOSYSTEM SERVICES", "CLIMATE TEMPERATURE trend",
                "CLIMATE TEMPERATURE status", "CLIMATE PRECIPITATION trend", "CLIMATE PRECIPITATION status",
                "CLIMATE WIND trend", "CLIMATE WIND status", "CLIMATE DROUGHT trend", "CLIMATE DROUGHT status",
                "CLIMATE EXTREME EVENTS trend", "CLIMATE EXTREME EVENTS status"]


SCHEMA_TYPES_SHORT = ["BIODIVERSITY", "SPECIES", "GENE", "ECOSYSTEM", "HABITAT", "FEEDING REGIME", "LIFE HISTORY", "DIVERSITY",
                    "POPULATION DENSITY", "POPULATION DISTRIBUTION", "POPULATION SIZE", "POPULATION DISTRIBUTION SPACE",
                    "POPULATION DISTRIBUTION ELEVATION", "POPULATION DISTRIBUTION TIME",
                    "CONSERVATION STATUS", "DRIVER", "FUNCTIONS",
                    "ECOSYSTEM SERVICES", "CLIMATE TEMPERATURE", "CLIMATE PRECIPITATION",
                    "CLIMATE WIND", "CLIMATE DROUGHT", "CLIMATE EXTREME EVENTS",]