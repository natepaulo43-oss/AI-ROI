# Contributing to AI ROI Prediction Tool

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/AI_ROI.git
   cd AI_ROI
   ```
3. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🛠️ Development Setup

### Backend Development

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # macOS/Linux

# Install dependencies
pip install -r backend/requirements.txt

# Run tests
pytest backend/tests/

# Start development server
cd backend
uvicorn app.main:app --reload
```

### Frontend Development

```bash
# Install dependencies
cd frontend
npm install

# Run development server
npm run dev

# Run linter
npm run lint

# Run type checking
npm run type-check
```

## 📝 Code Style

### Python (Backend)
- Follow PEP 8 style guide
- Use type hints for function parameters and return values
- Maximum line length: 100 characters
- Use docstrings for all functions and classes

### TypeScript/React (Frontend)
- Follow ESLint configuration
- Use TypeScript for all new files
- Use functional components with hooks
- Follow React best practices

## 🧪 Testing

### Backend Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov-report=html
```

### Frontend Tests
```bash
# Run unit tests
npm test

# Run with coverage
npm test -- --coverage
```

## 📦 Pull Request Process

1. **Update documentation** if you're changing functionality
2. **Add tests** for new features
3. **Ensure all tests pass** before submitting
4. **Update the README.md** if needed
5. **Write clear commit messages**:
   ```
   feat: Add new prediction endpoint
   fix: Resolve CORS issue in API
   docs: Update installation instructions
   ```

6. **Submit your PR** with:
   - Clear title and description
   - Reference to related issues
   - Screenshots (if UI changes)

## 🐛 Bug Reports

When reporting bugs, please include:
- **Description**: Clear description of the issue
- **Steps to reproduce**: Detailed steps to reproduce the bug
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: OS, Python version, Node version, browser
- **Screenshots**: If applicable

## 💡 Feature Requests

For feature requests, please include:
- **Use case**: Why this feature would be useful
- **Proposed solution**: How you envision it working
- **Alternatives**: Other solutions you've considered

## 📚 Documentation

- Update relevant documentation in `/docs`
- Keep README.md up to date
- Add inline comments for complex logic
- Update API documentation if endpoints change

## 🔍 Code Review

All submissions require review. We use GitHub pull requests for this purpose. Reviewers will check:
- Code quality and style
- Test coverage
- Documentation updates
- Performance implications
- Security considerations

## 🎯 Areas for Contribution

### High Priority
- [ ] Improve model accuracy with new features
- [ ] Add confidence intervals to predictions
- [ ] Implement user authentication
- [ ] Add more comprehensive tests

### Medium Priority
- [ ] Create mobile-responsive improvements
- [ ] Add data visualization enhancements
- [ ] Implement A/B testing framework
- [ ] Add internationalization (i18n)

### Good First Issues
- [ ] Improve error messages
- [ ] Add loading states
- [ ] Update documentation
- [ ] Fix minor UI bugs

## 📞 Questions?

If you have questions:
- Open a GitHub issue
- Check existing documentation in `/docs`
- Review the README.md

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing! 🎉**
