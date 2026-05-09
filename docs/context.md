You are a key team member. Help plan and implement the assignment based on the Milestone 2 requirements and the Milestone 1 resources. This file documents the scope, constraints, and what already exists in the workspace so another agent can continue the project without missing requirements.

The backend and frontend repos are not setup yet, please also help me.

## Source of truth (must follow)
- Primary requirements: [Milestone2_Requirement.md](Milestone2_Requirement.md)
	- All planning and implementation decisions must reference Milestone 2 first.
- Reference to Milestone 1 scope and outputs: [Milestone1_Requirement.md](Milestone1_Requirement.md)
- Code rules (must follow): [CodeRules.md](CodeRules.md)

## Milestone 2 requirements summary (high detail)
The system must be an online shopping website for cosmetics/beauty products, built on the Milestone 1 dataset and models.

### Task 1: Search for products
- Provide keyword search by brand name and/or description keywords.
- After search, the UI must:
	- show the count of matched items,
	- show a list of item previews,
	- allow clicking an item to see details and reviews.
- Search algorithm must support similar keyword forms (e.g., different casing or expanded brand form still returns the same results).

### Task 2: Create reviews and generate labels
- Users can add new reviews with title, description, and rating.
- The system must use one Milestone 1 model to predict a binary label (Buy / Not Buy) for the review.
- Show the predicted label to the user and allow the user to accept or override it.
- After confirmation, the review must be saved and accessible via a URL.

### Task 3: Recommend similar items
- When a product is selected, the system must show recommended similar products.
- You must define and implement a similarity measure (text features, embeddings, attributes, or other signals).

### Task 4: Additional functionality
- Implement at least one additional, fully working feature that improves usability or effectiveness.
- Must use technologies discussed in the course.

### Video demo requirements
- A single video (max 4 minutes) must demonstrate:
	1) browsing products,
	2) creating a review,
	3) label generation and review display,
	4) similar item recommendations,
	5) the additional functionality.

(You dont need to do the video for me)

### Submission packaging
- Must include all source code, website interface, and a README.txt with student info and run instructions.
- Include the demo video as .mp4. (Not yet, I will do later)
- If any required files exceed 50MB, upload them to OneDrive and include the link in README.txt.
- Zip a folder named after the group name and submit via Canvas.

### Local execution requirement
- The full system (frontend + backend + model inference) must run successfully on a local machine before submission.

## Existing assets from Milestone 1 (data + outputs)
All Milestone 1 assets live in [Milestone1_CodeBase/](Milestone1_CodeBase/). This folder contains:
- Raw dataset: [Milestone1_CodeBase/cosmetics_beauty_products_reviews.csv](Milestone1_CodeBase/cosmetics_beauty_products_reviews.csv)
- Stopword list: [Milestone1_CodeBase/stopwords_en.txt](Milestone1_CodeBase/stopwords_en.txt)
- Preprocessing output: [Milestone1_CodeBase/processed.csv](Milestone1_CodeBase/processed.csv)
- Vocabulary output: [Milestone1_CodeBase/vocab.txt](Milestone1_CodeBase/vocab.txt)
- Count vectors: [Milestone1_CodeBase/count_vectors.txt](Milestone1_CodeBase/count_vectors.txt)
- Unweighted embeddings: [Milestone1_CodeBase/unweighted_vectors.txt](Milestone1_CodeBase/unweighted_vectors.txt)
- TF-IDF weighted embeddings: [Milestone1_CodeBase/weighted_vectors.txt](Milestone1_CodeBase/weighted_vectors.txt)
- Milestone 1 notebooks and exports:
	- Task 1 preprocessing notebook and script: [Milestone1_CodeBase/task1.ipynb](Milestone1_CodeBase/task1.ipynb), [Milestone1_CodeBase/task1.py](Milestone1_CodeBase/task1.py)
	- Task 2/3 representation + modeling notebook and script: [Milestone1_CodeBase/task2_3.ipynb](Milestone1_CodeBase/task2_3.ipynb), [Milestone1_CodeBase/task2_3.py](Milestone1_CodeBase/task2_3.py)

### Milestone 1 output expectations (per rubric/teacher guidance)
- Task 1 outputs: `processed.txt` and `vocab.txt`.
- Task 2 outputs: `count_vectors.txt`, `unweighted_vectors.txt`, and `weighted_vectors.txt`.
- Task 3 outputs: only evaluation metrics (accuracy) are recorded in the notebook; no extra output files.
- `.py` files must be exported from the notebooks for plagiarism checking.

### Notes on the Milestone 1 code
- [Milestone1_CodeBase/task1.py](Milestone1_CodeBase/task1.py) implements the full preprocessing pipeline, writes `processed.csv` and `vocab.txt`, and documents each step in detail.
- [Milestone1_CodeBase/task2_3.py](Milestone1_CodeBase/task2_3.py) loads those outputs, builds count vectors and embedding-based vectors, and trains classifiers with 5-fold CV. It also explains why a specific pretrained embedding was chosen.
- No serialized model artifacts are currently stored. The Milestone 2 backend must export or re-train a chosen model and load it for inference.

## UI reference repo (must reuse design language)
The UI reference is the Vite + React project under [SampleUI/](SampleUI/). Reuse its components and theme to keep the same visual identity.

### Entry points and theme
- App bootstrap: [SampleUI/src/main.tsx](SampleUI/src/main.tsx) mounts the React app and imports [SampleUI/src/styles/index.css](SampleUI/src/styles/index.css).
- Global styles are layered in [SampleUI/src/styles/index.css](SampleUI/src/styles/index.css), which imports:
	- [SampleUI/src/styles/fonts.css](SampleUI/src/styles/fonts.css) (currently empty)
	- [SampleUI/src/styles/tailwind.css](SampleUI/src/styles/tailwind.css) (Tailwind setup)
	- [SampleUI/src/styles/theme.css](SampleUI/src/styles/theme.css) (CSS variables, typography defaults)
	- [SampleUI/src/styles/carousel.css](SampleUI/src/styles/carousel.css) (slick carousel base styles)

### Existing UI behavior in the reference app
- Main page and state handling: [SampleUI/src/app/App.tsx](SampleUI/src/app/App.tsx)
	- Uses local sample product data with brand, name, rating, reviews, price, description.
	- Implements text search over product name and brand.
	- Shows a product grid and opens a full-screen product detail panel on click.
	- Computes similar products based on same brand and shows them in the detail view.

- Header and search bar: [SampleUI/src/app/components/Header.tsx](SampleUI/src/app/components/Header.tsx)
	- Sticky header with nav links and a search input.
	- Displays the number of matching items when a search query is present.
	- Includes placeholder icons for wishlist, profile, and cart.

- Product grid and cards:
	- Grid layout: [SampleUI/src/app/components/ProductGrid.tsx](SampleUI/src/app/components/ProductGrid.tsx)
	- Product card: [SampleUI/src/app/components/ProductCard.tsx](SampleUI/src/app/components/ProductCard.tsx)
		- Uses [SampleUI/src/app/components/figma/ImageWithFallback.tsx](SampleUI/src/app/components/figma/ImageWithFallback.tsx) to handle broken images.
		- Shows brand, name, rating stars, review count, and price.

- Product detail view and review UI:
	- Detail panel: [SampleUI/src/app/components/ProductDetail.tsx](SampleUI/src/app/components/ProductDetail.tsx)
	- Includes:
		- Product imagery, rating summary, price, and description.
		- Similar products carousel using `react-slick`.
		- A review form with rating selection, title, description, and a displayed AI prediction.
		- A local-only submission list rendered beneath the form (no persistence in this reference).

### Component library inventory
The reference project includes a rich set of reusable UI primitives under:
- [SampleUI/src/app/components/ui/](SampleUI/src/app/components/ui/)
These are Radix-based UI components and utility helpers (for example, [SampleUI/src/app/components/ui/utils.ts](SampleUI/src/app/components/ui/utils.ts)). Reuse these components to keep styling consistent.

### Build tooling
- Vite + React + Tailwind setup: [SampleUI/vite.config.ts](SampleUI/vite.config.ts)
- Dependencies and scripts: [SampleUI/package.json](SampleUI/package.json)
- Basic run instructions: [SampleUI/README.md](SampleUI/README.md)

## Backend implementation requirement
- The backend must be implemented with Flask API.
- It must expose endpoints for search, product details, review submission, and prediction.
- It must load (or compute) the chosen Milestone 1 model and run inference for new reviews.

## Collaboration expectation
Work as a proactive, detail-oriented team member. Use this file as a handover document that explains the requirements, the data and code already present, and the UI assets that must be reused.
