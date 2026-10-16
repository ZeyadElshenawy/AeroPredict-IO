# FlightOps ML — Problem Definition

## Business problem
An operations team needs to prioritize flights that may experience
significant departure delays.

## Intended user
A hypothetical airline or airport operations analyst.

## Prediction target
A departure delay of at least 15 minutes.

## Prediction horizon
Two hours before scheduled departure.

## Initial data source
U.S. Bureau of Transportation Statistics (BTS),
Reporting Carrier On-Time Performance data.

## Intended prediction unit
One eligible scheduled flight.

## Proposed output
A delay-risk probability, a risk category, and an explanation
of the main contributing features.

## Initial scope
Historical binary classification for flights with eligible,
observed departure outcomes.

Cancellations and diversions will be audited and reported separately
until an explicit policy is established.

## Known limitations
- Historical data does not automatically establish feature
  availability at prediction time.
- The initial system is a portfolio prototype, not an operational
  airline deployment.
- Model performance and business value have not yet been measured.

## Open questions
- Which reporting months provide sufficiently complete data?
- Which fields are available at the prediction cutoff?
- How should cancellations and diversions be handled?
- What operational action would follow a high-risk prediction?
- What are the costs of false positives and false negatives?