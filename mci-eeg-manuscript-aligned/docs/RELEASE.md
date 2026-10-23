# GitHub and Zenodo release checklist

This package is prepared locally. No repository, upload or DOI has been created.

1. Finalize the analysis protocol and document the validation status of the release.
2. Upload only this clean source tree to a new GitHub repository. Inspect the
   actual staged files; `.gitignore` cannot remove already tracked private files.
   No local dataset, output directory, absolute patient paths or credentials.
3. Add the actual creators/ORCIDs and correct affiliation metadata. Confirm the
   MIT license is acceptable to all code owners. Do not invent author metadata.
4. Run tests in a clean environment. Save exact software versions and tag a
   version, for example `v0.1.0`, after review. Record the Git commit with results.
5. Sign into Zenodo, connect the GitHub account, enable the repository and create
   a GitHub release. Check the imported files and creator/license metadata.
   Archive the release and obtain its real version DOI. A concept DOI refers to
   all versions; cite the version DOI used for the reported results.
6. Record the repository URL, version and DOI in the paper, identifying the
   version used to generate the reported results.

Official sources checked for this package:

- [Aperture Neuro author guidance](https://apertureneuro.org/for-authors)
- [Aperture Neuro editorial/about policy](https://apertureneuro.org/about)
- [Zenodo GitHub integration documentation](https://help.zenodo.org/docs/github/)

The journal asks authors to explain access conditions and provide repository
links, versions and persistent identifiers where available. Its open-source
encouragement should not be paraphrased as an unconditional legal requirement
that every code/material item must be made public. Review the current policy and
applicable restrictions at submission.

## Availability statement template — fill only after actual release

> The analysis code is available at [REPOSITORY URL], version [VERSION],
> archived at [VERSION DOI]. Its validation status is described in the repository.
> Individual-level
> EEG and clinical data are not publicly available because [specific applicable
> consent, ethics and privacy restrictions]. Access requests are evaluated by
> [responsible institution/contact] under [eligibility, review procedure and
> data-use agreement conditions]. A synthetic dataset generator is provided for
> software testing and contains no participant data.

Do not insert a fictional DOI or claim that restricted access is approved without
institutional confirmation. State realistic access conditions, not just "upon
request". Update any AI-use statement to accurately cover AI-assisted coding and
methodological review where applicable; authors retain responsibility for checking
the implementation and scientific claims.
