# Sources and indicator definitions

All historical observations are requested from the official World Bank Indicators API:
https://api.worldbank.org/v2/country/{COUNTRY}/indicator/{INDICATOR}?date={START}:{END}&format=json

Country ISO3 API identifiers: Mexico MEX; Brazil BRA; Argentina ARG; Colombia COL; Chile CHL; Peru PER; Venezuela VEN; Ecuador ECU; Uruguay URY. Brazil is included as a Latin American comparison country and is not a Spanish-speaking country.

| Indicator | World Bank code | Units | Interpretation |
|---|---|---|---|
| Life expectancy at birth | `SP.DYN.LE00.IN` | years | Expected years lived under current mortality patterns. |
| Unemployment, national estimate | `SL.UEM.TOTL.NE.ZS` | % of labor force | Unemployment share using national estimates. |
| GDP per capita | `NY.GDP.PCAP.CD` | current US dollars per person | GDP divided by midyear population; not average wealth or household income. |
| Crude birth rate | `SP.DYN.CBRT.IN` | births per 1,000 people per year | Rate, not total number of births. |
| Net migration | `SM.POP.NETM` | people | Net migration measure, not an emigration count. |
| Intentional homicide rate | `VC.IHR.PSRC.P5` | per 100,000 people | Intentional homicides per 100,000 population. |
| Consumer price inflation | `FP.CPI.TOTL.ZG` | annual % | Annual percentage change in consumer prices. |
| Central government debt | `GC.DOD.TOTL.GD.ZS` | % of GDP | Central government debt relative to GDP. |
| Total reserves including gold | `FI.RES.TOTL.CD` | current US dollars | Total reserves including gold. |
| External balance on goods and services | `NE.RSB.GNFS.ZS` | % of GDP | External balance on goods and services relative to GDP. |

The app requests calendar years from current year minus 70 through current year minus 1, inclusive. The API may return fewer non-null observations. Coverage and count are computed from returned values. Missing observations are not imputed or replaced with random data.

Retrieval timestamps are generated when the app runs. Indicator metadata and observations are provided by the World Bank; consult the linked API and World Bank indicator pages for the latest metadata and methodology.
