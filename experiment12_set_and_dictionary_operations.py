# EXPERIMENT 12 - Set and Dictionary Operations
# AIM: Analyse course enrolments and student marks using sets and dictionaries.

from collections import Counter, defaultdict

print("University Course Enrolment Analysis: Sets and Dictionaries")
print("=" * 60)

# ---------- Sets: course enrolments ----------
python_course = {"Anu", "Bala", "Chitra", "Dinesh", "Esha", "Farhan"}
ml_course = {"Chitra", "Dinesh", "Gokul", "Harini", "Esha"}
big_data = {"Bala", "Esha", "Harini", "Ishaan"}

print("\nPython course:", sorted(python_course))
print("ML course:", sorted(ml_course))
print("Big Data:", sorted(big_data))
print("Union (all students):", sorted(python_course | ml_course | big_data))
print("Intersection (Python & ML):", sorted(python_course & ml_course))
print("Difference (Python only, not ML):", sorted(python_course - ml_course))
print("Symmetric difference (Python ^ ML):", sorted(python_course ^ ml_course))
print("In all three courses:", sorted(python_course & ml_course & big_data))
print("Is {'Esha','Bala'} subset of Python?:",
      {"Esha", "Bala"}.issubset(python_course))
print("Are ML and Big Data disjoint?:", ml_course.isdisjoint(big_data))

ml_course.add("Jaya")
ml_course.discard("Gokul")
print("ML after add('Jaya'), discard('Gokul'):", sorted(ml_course))

duplicates = ["Anu", "Bala", "Anu", "Chitra", "Bala", "Anu"]
print("Unique names from a list with duplicates:", sorted(set(duplicates)))

# ---------- Dictionaries: marks ----------
marks = {
    "Anu": {"Python": 88, "ML": None},
    "Bala": {"Python": 72, "BigData": 81},
    "Chitra": {"Python": 95, "ML": 91},
    "Dinesh": {"Python": 64, "ML": 70},
}

# Update an existing nested entry and insert a new student
marks["Anu"]["ML"] = 79
marks["Esha"] = {"Python": 85, "ML": 88, "BigData": 90}

print("\nStudent marks dictionary:")
for student, subjects in marks.items():
    print(f" {student:<7} {subjects}")

print("Keys:", list(marks.keys()))
print("Chitra's Python mark:", marks["Chitra"]["Python"])
print("get() for missing key:", marks.get("Zara", "Not enrolled"))

# Dictionary comprehension: calculate each student's average
averages = {
    student: round(sum(scores.values()) / len(scores), 2)
    for student, scores in marks.items()
}
print("Average per student:", averages)

topper = max(averages, key=averages.get)
print("Topper:", topper, averages[topper])

# defaultdict groups marks by subject
subject_scores = defaultdict(list)
for subjects in marks.values():
    for subject, mark in subjects.items():
        if mark is not None:
            subject_scores[subject].append(mark)

print("\nSubject-wise statistics:")
for subject, scores in subject_scores.items():
    print(f" {subject:<8} count={len(scores)} "
          f"avg={sum(scores)/len(scores):.2f} max={max(scores)}")

removed = marks.pop("Dinesh")
print("\nRemoved Dinesh ->", removed, "| students left:", len(marks))
marks.update({"Farhan": {"Python": 77}})
print("After update() with Farhan:", sorted(marks))

# Counter: word-frequency analysis of course feedback
feedback = (
    "good course good faculty excellent labs good projects excellent support"
)
frequency = Counter(feedback.split())
print("\nWord frequency of feedback:", dict(frequency))
print("Most common 2 words:", frequency.most_common(2))
print("Sorted by value (desc):", dict(sorted(
    frequency.items(), key=lambda item: item[1], reverse=True
)))
