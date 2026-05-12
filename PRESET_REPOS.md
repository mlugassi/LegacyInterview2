# 📦 Preset GitHub Repositories for Debugging Challenges

This file lists all available preset repositories that are pre-configured and tested for the Legacy Code Challenge system.

---

## ✅ Currently Integrated (Available in Dropdown)

These repositories are already available in the GUI dropdown menu:

### 1. **boltons** ⭐ RECOMMENDED
- **URL:** https://github.com/mahmoud/boltons
- **Description:** Comprehensive utility library with functions for strings, lists, dicts, math, iterators
- **Why it's great:** Clean, well-documented code with many primitive-type functions
- **Best for:** All difficulty levels

### 2. **more-itertools**
- **URL:** https://github.com/more-itertools/more-itertools
- **Description:** Advanced iterator utilities beyond the standard library
- **Why it's great:** Pure Python, many simple functions with clear logic
- **Best for:** Medium difficulty (nesting-level 3-4)

### 3. **toolz**
- **URL:** https://github.com/pytoolz/toolz
- **Description:** Functional programming utilities for Python
- **Why it's great:** Short, composable functions with minimal dependencies
- **Best for:** Low to medium difficulty

### 4. **funcy**
- **URL:** https://github.com/Suor/funcy
- **Description:** Collection manipulation utilities
- **Why it's great:** Simple, readable code with good test coverage
- **Best for:** All levels

### 5. **iteration_utilities**
- **URL:** https://github.com/MSeifert04/iteration_utilities
- **Description:** Fast iterator helpers and utilities
- **Why it's great:** Well-structured with helper function chains
- **Best for:** Medium to high difficulty (nesting-level 4-5)

### 6. **algorithms** (python-algorithms)
- **URL:** https://github.com/keon/algorithms
- **Description:** Classic algorithms implemented in Python
- **Why it's great:** Educational code, clear implementations
- **Best for:** High difficulty (nesting-level 5-6)

### 7. **binaryornot**
- **URL:** https://github.com/audreyr/binaryornot
- **Description:** Detect if a file is binary or text
- **Why it's great:** Small, focused library with file handling
- **Best for:** Low difficulty (simple projects)

### 8. **humanize**
- **URL:** https://github.com/python-humanize/humanize
- **Description:** Human-readable formatting for numbers, dates, file sizes
- **Why it's great:** String manipulation with edge cases
- **Best for:** Medium difficulty

### 9. **python-dateutil**
- **URL:** https://github.com/dateutil/dateutil
- **Description:** Powerful date/time manipulation library
- **Why it's great:** Complex logic with many edge cases
- **Best for:** High difficulty (complex logic)

### 10. **inflection**
- **URL:** https://github.com/jpvanhal/inflection
- **Description:** String transformations (pluralize, camelize, etc.)
- **Why it's great:** Short functions, string manipulation
- **Best for:** Low to medium difficulty

---

## 🔍 Additional Recommended Repositories (To Add)

These repositories would work well but aren't yet added to the dropdown:

### String/Text Processing
- **python-slugify** - https://github.com/un33k/python-slugify
- **ftfy** - https://github.com/rspeer/python-ftfy
- **arrow** - https://github.com/arrow-py/arrow

### Data Structures
- **sortedcontainers** - https://github.com/grantjenks/python-sortedcontainers
- **bidict** - https://github.com/jab/bidict
- **cachetools** - https://github.com/tkem/cachetools

### Math/Numbers
- **num2words** - https://github.com/savoirfairelinux/num2words
- **pyparsing** (examples folder) - https://github.com/pyparsing/pyparsing

### File/Path Utilities
- **pathlib2** - https://github.com/jazzband/pathlib2
- **scandir** - https://github.com/benhoyt/scandir

---

## 🚫 Not Recommended

Avoid these types of repositories:

❌ **Web frameworks** (Django, Flask) - Too complex, class-heavy
❌ **ML libraries** (NumPy, TensorFlow) - External dependencies, compiled code
❌ **GUI frameworks** (tkinter, PyQt) - Event-driven, hard to test
❌ **Network libraries** (requests, urllib3) - I/O operations, mocking needed
❌ **Database ORMs** (SQLAlchemy) - Too many abstractions

---

## 🎯 Selection Criteria

A good repository should have:

✅ **Pure Python** - No C extensions or compiled code
✅ **Primitive types** - Functions work with int, str, list, dict
✅ **Call chains** - Functions that call other helper functions
✅ **Minimal dependencies** - Stdlib or 1-2 simple deps
✅ **Clear logic** - Not overly abstract or metaprogramming-heavy
✅ **Good structure** - Organized modules, not one giant file

---

## 📖 How to Use

### In GUI Mode:
1. Launch: `python challenge.py`
2. Select "Choose from preset list"
3. Pick a repository from the dropdown
4. Click "Start Challenge"

### In CLI Mode:
```bash
# List all available repos
python challenge.py --list-repos

# Use a preset repo
python challenge.py https://github.com/mahmoud/boltons --name "Alice"

# Custom repo
python challenge.py https://github.com/YOUR/REPO --name "Alice" --nesting-level 4
```

---

## 🔧 Adding New Presets

To add a new preset repository:

1. Test it first with a custom URL
2. Edit `student_interface.py`
3. Add to `PRESET_REPOS` dictionary:
   ```python
   "Short Name - Description": "https://github.com/user/repo",
   ```
4. Save and restart the application

---

**Last Updated:** May 11, 2026
**Total Preset Repos:** 10
