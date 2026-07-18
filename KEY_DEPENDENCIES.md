The following is a list of the key dependencies that are absolutely necessary to understand in detail in order to contribute to this project. Note that this is a mono-repo, containing several projects, and we are starting a new project entirely (new approach under the approaches folder). It may be useful to examine other projects inside the mono-repo to understand how to use the key dependencies, but this project will effectively be separate from the previous projects contained. These key dependencies are also contained in @pyproject.toml

- relational_structs
- bilevel-planning
- kinder-models
- kindergarden (imported as kinder)

some important substrates are (read these, do not modify):
- structs.py: core types - RelationalAbstractState, GroundSkill, LiftedSkill, SesameModels
- pddl.py: skeleton-related PDDL types
- objects.py: obejct/type primatives

{ClutteredRetrieval2D,ClutteredStorage2D,Motion2D,Obstruction2D,StickButton2D}.py + .md
- the five 2D kinder environments SPECTRE will be evaluated on. Ensure you understand the details of these environments in depth, such as their predicates and operators, etc.

also note that the dependencies have been installed in a virtual environment contained in .venv, which can be activated with source .venv/bin/activate; python should NEVER be called by itself globally, and you should ALWAYS use the virtual environment. This project is managed using uv as the package manager.