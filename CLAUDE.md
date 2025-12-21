# CLAUDE.md

## REQUIRED: Session Start Checklist

At the start of a new session, you MUST do these things before working on a new task from the user:

1. Scan backend/migrations/* and /scripts/* to understand the full project schema and how components are run
2. Review /docs/* for architecture patterns and conventions
3. Run `git log --oneline -20` to understand recent work context, then `git status` to check for uncommitted changes
4. IF continuing previous work: Summarize last session's progress
5. Explain the development workflow to the user including monitoring the Node.js backend, accessing Postgres, what ports everything is running on, etc.


## Tool Usage Efficiency
- Batch related file reads in a single message
- Use Grep/Glob for targeted searches before using Task tool
- Avoid re-reading unchanged files in same session
- Run multiple bash commands in parallel when possible

## Task Priority Assessment
1. **Critical** - Security issues, breaking bugs → immediate action
2. **Requested** - User-specified features → implement exactly as requested
3. **Suggested** - Code quality improvements → propose but don't implement unprompted
4. **Documentation** - Update only when code changes require it

## Communication Templates

**Finding an issue:**
"Found [issue] in [file:line]. This could cause [impact]. Fix with [solution]?"

**Completing a task:**
"✓ [Task]. [Brief what was done]. Tests: [pass/fail/none found]"

**Needing clarification:**
"[Requirement] could mean [option A] or [option B]. Which would you prefer?"

**Error reporting:**
- Error: [what failed]
- Location: [file:line]
- Cause: [why it failed]
- Fix: [suggested solution]
- Impact: [what this blocks]


## README.md Policy
- NEVER read or write README.md unless the user specifically tells you to.

## Technical Guidelines

### Before writing code:
- Check existing patterns in neighboring files
- Verify required libraries are already in use
- Match existing code style and conventions

### When implementing:
- Write small, focused functions
- Use meaningful variable names
- Handle errors gracefully
- Follow security best practices (never commit secrets)

## Policies & Constraints

### Explicit Don'ts
- DON'T commit to git unless explicitly asked - NEVER proactively create commits
- DON'T refactor working code without permission
- DON'T create example/demo/test files unless requested
- DON'T implement "nice-to-have" features not requested
- DON'T add code comments unless complex logic requires it
- DON'T use fluffy or ingratiating language when speaking to the user


### Naming Quick Reference
- **Database**: snake_case (tables, columns, indexes)
- **Java**: PascalCase classes, camelCase methods/fields
- **JavaScript**: PascalCase components, camelCase functions/variables
- **API**: camelCase JSON properties, kebab-case URL paths
- **URL parameters**: camelCase (e.g., ?apiKey=abc123&userId=456)

### Conflict Resolution Guidelines

**When user request conflicts with best practices:**
- Explain the concern briefly
- Offer the best-practice alternative
- Implement user's choice if they insist

**When existing code doesn't follow conventions:**
- Follow existing patterns in that file/module
- Only refactor if user requests it

**During migrations (e.g., Node to Java):**
- Document migration status in memory bank
- Keep deprecated code until user confirms removal

## Default Preferences

### Git Workflow
- Use meaningful commit messages
- Never commit sensitive information
- Back up before major refactoring

## Important Reminders
- Prefer editing existing files over creating new ones
- Never create files unless absolutely necessary
- Report progress on complex tasks using TodoWrite tool
- Be direct and efficient in communication