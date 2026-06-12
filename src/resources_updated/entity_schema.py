SCHEMA_BIODIV_FULL = """
Abiotic entity (A non-living natural physical object or component of the environment)
        Abiotic assemblance
                glacial deposits
                glacial moraine complex
                sediment layers
                soil horizons
                watershed network (abiotic components)
        Abiotic physical object
                boulder
                glacier
                permafrost layer
                rock outcrop
                scree
                snowpack
                soil
                talus slope
        Aquatic landform
                lake
                river
                spring
        Terrestrial landform
                cave
                high mountain environments
                moraine
                mountain
                mountains
                valley
                
Abiotic process (A physical or chemical change in the environment)
        Atmospheric process
                atmospheric circulation
                climate change
                climate warming
                climatic changes
                condensation
                convection
                freeze-thaw
                global warming
                heavy snowfall
                orographic uplift
                temperature rise
                thermal inversion
                wind drift
        Cryospheric process
                glacial melt
                glacier flow
                permafrost thaw
                snow accumulation
        Geological process
                orogeny
                weathering
        Geormorphological process
                avalanche
                erosion
                landslide
                river incision
                river meandering
                sedimentation
        Soil process
                soil leaching
                soil physical degradation
                soil sealing
            
Abiotic property (A measurable physical or chemical attribute)
        Atmospheric property
                climatic conditions
                cold climate
                mean global temperature
                mean summer temperature of year t-1
                total precipitation during winter
                weather condition
                wind speed
        Chemical property
                hardness
                salinity
        Cryospheric property
                snow cover
                snow depth
        Environmental property
                abiotic environmental factors
                extreme environmental conditions
        Physical property
                albedo
                conductivity
                low temperature
                radiation
                temperature
        Soil property
                soil moisture content
                soil pH

Anthropogenic entity (A human-made object)
        Built infrastructure
                building
                dam
                fence
                hydropower infrastructure
        Social construct
                land management system
                protected area designation (infrastructure)
                conservation policy
                customary law
                land tenure regime
                management plan
                

                
Anthropogenic process (An activity or action performed by humans that affects socioecological systems)
        Intellectual /governance process
                community forest management
                conservation planning
                forest ecology
                governing
                managing
                policy implementation
                surveys
        Physical / land-use process
                competition
                forestry
                grazing (as a human activity)
                habitat destruction
                human migration
                hunting
                irrigation
                land-use change
                logging
                mining
                road construction
                terracing
                tourism
                
Anthropogenic property (A measurable social, economic or governance attribute)
        Demographic property
                human population density
                population (human)
        Economic property
                agricultural intensity
                GDP per capita
                household income
                tourism pressure
        Educational property
                literacy rate
        Governance property
                governance effectiveness
                management category
        Legal property
                land tenure type
                ownership
        Material property
                infrastructure density
                
Antropoggenic social institution (Non-material human constructs such as laws, policies, norms, governance structures and knowledge systems)
        Institutional arrangement
                conservation policy
                customary law
                land tenure regime
                management plan
                protected area designation (legal)
                traditional knowledge
        
Biotic entity (A living organism or taxon at the individual/species level)
                Community
                epiphyte assemblage
                fauna
                mixed broadleaf forest
                pollinator community
                predator-prey system
                riparian vegetation
                soil microbial community
                ungulate guild
                vegetation
                vegetation layers
        Ecosystem
                alpine grassland
                biocenosis
                conifer forest
                ecosystem
                shrubland
                wetland ecosystem
        Population
                wolf population
                population (biology)
        Organism
                animals
                fir tree
                herbivorous species
                high-mountain species
                juniper tree
                organism
                ruminant
        Taxon
            alpine sagebush
            amniote
            bumblebee species
            frog species
            lichen
            mammal
            moss species
            pika
            plants
            rhododendron
            snow leopard
            species
            spruce
            tatra chamois
            tatra chamois
            vertebrate
            Vulpes vulpes
            yak
        Organic part or material
            food resource
            organ
            liver
            stem
            leaf
            carcass

Biotic process (A biological action or interaction performed by or involving living organisms)
        Behavioral process
                animal migration
                migration
        Ecosystem process
                ecosystem service
                succession*
                nutrient cycling
                biogeochemical cycling
                primary succession
        Biotic interactions
                competition
                parasitism
                pollination
                seed dispersal
                seed rain
        Physiological process
                decomposition
                flowering
                germination
                reproduction
                symbiosis
        Population process
                disappearance of species
                population decrease
                population differentiation
                population dynamic
                population increase
        Trophic process
                grazing
                herbivory
                predation
        Phenological process
                phenology
                
Biotic property (A trait, attribute or measurable characteristic of organisms or biotic assemblages)
        Aggregate trait
                biodiversity
                forest biodiversity
                species richness
        Biogeographical trait
                endemism
                home range
                ranges of species
        Conservation trait
                habitat condition
        Ecosystem trait
                canopy height
        Functional trait
                biomass
        General trair
                growth rate
        Genetic trait
                genetic diversity
        Physiological trait
                nutrient uptake efficiency
                wood density
        Population trait
                population growth rate
                population status
                population trend
        Species physical trait
                body size
                leaf area
        Phenological trait
                duration of vegetation period

Concept (An abstract concept or theoretical construct used in analysis or discourse (not directly a material object or process).
        Abstract concept
                biodiversity
                carrying capacity
                climate
                conservation status
                conservation value
                environment
                resilience
                scenarios
                trends
                vulnerability
        
Spatial entity (A named or operational geographic extent or geometric unit used for mapping or aggregation)
        Administrative region
                buffered zone
                district
                Poland
                protected area
                Tatra National Park
                Zakopane
        Conceptial spatial unit
                grid cell
                polygon
                slope
                transect
        Natural region
                Arctic region
                cold climatic zones
                elevational belt
                vegetation zones
                watershed
        
Spacial property (A geometric or positional descriptor of an entity)
        Geometric property
                area
                spatial extent
        Topographic property
                elevation
                slope
        Topological property
                aspect
                connectivity
                distance to river
                distance to road
                fragmentation index

Temporal entity (A named or identifiable time period or phase)
        Geological era
                Holocene
                Pleistocene
        Seasonal entity
                breeding season#
                dry season
                fire season
                growing season
                monsoon season
                season
                spring
                winter

Temporal property (A temporal descriptor or metric)
        Cyclical property
                annual
                decadal
                diurnal
                diel
                duration
                early-season
                interannual variability
                late-season
        Temporal scale
                long-term
                phenological timing
                short term
Qualitative property (A non-numeric attribute that describes a quality or category)
        high/low
        present/absent
        increasing/decreasing
        good/poor
        big/small
        fast/slow
        unstable/stable

Quantitative property (A measurable attribute that can be expressed as a quantity)
        16%
        100 meters above the sea level
        3000 km
        15 species
        half of a populations
"""

SCHEMA_BIODIV = """

Abiotic entity (A non-living natural physical object or component of the environment)
        glacial deposits
        sediment layers
        soil horizons
        watershed network (abiotic components)
        glacier
        permafrost layer
        soil
        talus slope
        lake
        river
        spring
        cave
        high mountain environments
        valley

Abiotic process (A physical or chemical change in the environment)
        atmospheric circulation
        climate change
        freeze-thaw
        global warming
        heavy snowfall
        orographic uplift
        temperature rise
        thermal inversion
        wind drift
        glacial melt
        permafrost thaw
        snow accumulation
        orogeny
        weathering
        avalanche
        erosion
        landslide
        river incision
        river meandering
        sedimentation
        soil leaching
        soil physical degradation
        soil sealing

Abiotic property (A measurable physical or chemical attribute)
        climatic conditions
        cold climate
        mean global temperature
        mean summer temperature of year t-1
        total precipitation during winter
        weather condition
        wind speed
        salinity
        snow cover
        snow depth
        abiotic environmental factors
        extreme environmental conditions
        albedo
        conductivity
        low temperature
        radiation
        soil moisture content
        soil pH

Anthropogenic process (An activity or action performed by humans that affects socioecological systems)
        community forest management
        conservation planning
        forest ecology
        governing
        managing
        policy implementation
        surveys
        competition
        forestry
        grazing (as a human activity)
        habitat destruction
        human migration
        hunting
        irrigation
        land-use change
        logging
        mining
        road construction
        terracing
        tourism

Anthropogenic property (A measurable social, economic or governance attribute)
        human population density
        population (human)
        agricultural intensity
        GDP per capita
        household income
        tourism pressure
        literacy rate
        governance effectiveness
        management category
        land tenure type
        ownership
        infrastructure density

Anthropogenic entity (A human-made object)
        building
        dam
        fence
        hydropower infrastructure
        land management system
        protected area designation (infrastructure)
        conservation policy
        customary law
        land tenure regime
        management plan

Biotic entity (An assemblage of organisms functioning as a unit (community, guild, ecosystem) or a living organism or taxon at the individual/species level)
        epiphyte assemblage
        fauna
        mixed broadleaf forest
        pollinator community
        predator-prey system
        riparian vegetation
        soil microbial community
        ungulate guild
        vegetation layers
        alpine grassland
        biocenosis
        conifer forest
        ecosystem
        shrubland
        wetland ecosystem
        wolf population
        animals
        fir tree
        herbivorous species
        high-mountain species
        juniper tree
        organism
        alpine sagebush
        amniote
        bumblebee species
        frog species
        lichen
        mammal
        moss species
        pika
        plants
        rhododendron
        snow leopard
        species
        spruce
        tatra chamois
        vertebrate
        Vulpes vulpes
        yak
        food resource
        organ
        liver
        stem
        leaf

Biotic process (A biological action or interaction performed by or involving living organisms)
        animal migration
        ecosystem service
        succession
        nutrient cycling
        biogeochemical cycling
        competition
        parasitism
        pollination
        seed dispersal
        seed rain
        decomposition
        flowering
        germination
        reproduction
        symbiosis
        disappearance of species
        population decrease
        population differentiation
        population dynamic
        grazing
        herbivory
        predation
        phenology

Biotic property (A trait, attribute or measurable characteristic of organisms or biotic assemblages)
        biodiversity
        forest biodiversity
        species richness
        endemism
        home range
        ranges of species
        habitat condition
        canopy height
        biomass
        growth rate
        genetic diversity
        nutrient uptake efficiency
        wood density
        population growth rate
        population status
        population trend
        body size
        leaf area
        duration of vegetation period

Concept (An abstract concept or theoretical construct used in analysis or discourse (not directly a material object or process).
        biodiversity
        carrying capacity
        climate
        conservation status
        conservation value
        environment
        resilience
        scenarios
        trends
        vulnerability

Spatial entity (A named or operational geographic extent or geometric unit used for mapping or aggregation)
        buffered zone
        district
        Poland
        protected area
        Tatra National Park
        Zakopane
        grid cell
        polygon
        slope
        transect
        Arctic region
        cold climatic zones
        elevational belt
        vegetation zones
        watershed

Spacial property (A geometric or positional descriptor of an entity)
        area
        spatial extent
        elevation
        slope
        aspect
        connectivity
        distance to river
        distance to road
        fragmentation index

Temporal entity (A named or identifiable time period or phase)
        Holocene
        Pleistocene
        breeding season
        dry season
        fire season
        growing season
        monsoon season
        season
        spring
        winter

Temporal property (A temporal descriptor or metric)
        annual
        decadal
        diurnal
        diel
        duration
        early-season
        interannual variability
        late-season
        long-term
        phenological timing
        short term

Quantitative property (A measurable attribute that can be expressed as a quantity)
        rate
        mean
        percentage
        index
        number
        count
        density
        frequency
        magnitude
    
Qualitative property (A non-numeric attribute that describes a quality or category)
        high/low
        present/absent
        increasing/decreasing
        good/poor
        big/small
"""

SCHEMA_BIODIV_SHORT = """
ABIOTIC ENTITY (DEFINITION: A non-living natural physical object or component of the environment, e.g., rock, glacier, soil, river)

ABIOTIC PROCESS (DEFINITION: A physical or chemical change in the environment (e.g., erosion, landsliding, glacial melt).

ABIOTIC PROPERTY (DEFINITION: A measurable physical or chemical attribute)

ANTHROPOGENIC ENTITY (DEFINITION: A human-made object, material or social construct, e.g., road, dam, policy)

ANTHROPOGENIC PROCESS (DEFINITION: An activity or action performed by humans that affects socioecological systems, e.g., grazing, mining, policy implementation)

ANTHROPOGENIC PROPERTY (DEFINITION: A measurable social, economic or governance attribute, e.g., population density, land tenure type)

BIOTIC ENTITY (DEFINITION: An assemblage of organisms functioning as a unit or a living organism or taxon at the individual/species level, e.g., snow leopard, juniper species)

BIOTIC PROCESS (DEFINITION: A biological action or interaction performed by or involving living organisms, e.g., pollination, predation, succession)

BIOTIC PROPERTY (DEFINITION: A trait, attribute or measurable characteristic of organisms or biotic assemblages, e.g., biomass, species richness)

SPATIAL ENTITY (DEFINITION: A named or operational geographic extent or geometric unit used for mapping or aggregation, e.g., watershed, grid cell, elevation belt)

SPATIAL PROPERTY (DEFINITION: A geometric or positional descriptor of an entity, e.g., elevation, slope, aspect, distance)

TEMPORAL ENTITY (DEFINITION: A named or identifiable time period or phase, e.g., breeding season, monsoon, Holocene)

TEMPORAL PROPERTY (DEFINITION: A temporal descriptor or metric, e.g., annual, decadal, phenological timing)

QUANTITATIVE PROPERTY (DEFINITION: A measurable attribute that can be expressed as a quantity, e.g., rate, mean, percentage, index)

QUALITATIVE PROPERTY (DEFINITION: A non-numeric attribute that describes a quality or category, e.g., high/low, present/absent, increasing/decreasing, good/poor)
"""

SCHEMA_BIODIV_LIST = [
    "ABIOTIC ENTITY",
    "ABIOTIC PROCESS",
    "ABIOTIC PROPERTY",
    "ANTHROPOGENIC ENTITY",
    "ANTHROPOGENIC PROCESS",
    "ANTHROPOGENIC PROPERTY",
    "BIOTIC ENTITY",
    "BIOTIC PROCESS",
    "BIOTIC PROPERTY",
    "CONCEPT",
    "SPATIAL ENTITY",
    "SPATIAL PROPERTY",
    "TEMPORAL ENTITY",
    "TEMPORAL PROPERTY",
    "QUANTITATIVE PROPERTY",
    "QUALITATIVE PROPERTY",
]