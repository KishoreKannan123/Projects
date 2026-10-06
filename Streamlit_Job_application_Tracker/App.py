import streamlit as st
import argparse
import psycopg2
import pandas as pd
from datetime import date


# ============================================================
# PostgreSQL configuration
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("--host", default="localhost")
parser.add_argument("--port", type=int, default=5433)
parser.add_argument("--database", default="postgres")
parser.add_argument("--user", default="postgres")
parser.add_argument("--password", required=True)

args, _ = parser.parse_known_args()

HOST = args.host
PORT = args.port
DATABASE = args.database
USER = args.user
PASSWORD = args.password


TABLE = "applications"


# ============================================================
# Database connection
# ============================================================

def get_connection():
    return psycopg2.connect(
        host=HOST,
        port=PORT,
        database=DATABASE,
        user=USER,
        password=PASSWORD
    )


# ============================================================
# Create table if it doesn't exist
# ============================================================

def initialize_database():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id SERIAL PRIMARY KEY,

            company TEXT NOT NULL,
            role TEXT NOT NULL,

            job_url TEXT,

            date_found DATE,
            date_applied DATE,

            status TEXT,

            location TEXT,
            salary TEXT,
            source TEXT,

            follow_up_date DATE,

            notes TEXT
        );
    """)

    conn.commit()

    cursor.close()
    conn.close()


# Initialize database
try:
    initialize_database()
except Exception as e:
    st.error(f"Failed to initialize database: {e}")
    st.stop()


# ============================================================
# Page
# ============================================================

st.title("Job Application Tracker")

STATUS_OPTIONS = [
    "Pending",
    "Interview",
    "Waiting for Result",
    "Offer",
    "Rejected"
]


# ============================================================
# ADD APPLICATION
# ============================================================


st.header("Add New Application")

# Initialize reset flag
if "reset_application_form" not in st.session_state:
    st.session_state.reset_application_form = False

# Reset form values after successful submission
if st.session_state.reset_application_form:
    st.session_state.company = ""
    st.session_state.role = ""
    st.session_state.job_url = ""
    st.session_state.location = ""
    st.session_state.salary = ""
    st.session_state.source = ""
    st.session_state.notes = ""
    st.session_state.status = "Pending"
    st.session_state.date_found = date.today()
    st.session_state.date_applied = date.today()
    st.session_state.follow_up_date = None

    st.session_state.reset_application_form = False


with st.form("application_form"):

    company = st.text_input(
        "Company",
        key="company"
    )

    role = st.text_input(
        "Role",
        key="role"
    )

    job_url = st.text_input(
        "Job URL",
        key="job_url"
    )

    date_found = st.date_input(
        "Date Found",
        value=date.today(),
        key="date_found"
    )

    date_applied = st.date_input(
        "Date Applied",
        value=date.today(),
        key="date_applied"
    )

    status = st.selectbox(
        "Status",
        STATUS_OPTIONS,
        key="status"
    )

    location = st.text_input(
        "Location",
        key="location"
    )

    salary = st.text_input(
        "Salary",
        key="salary"
    )

    source = st.text_input(
        "Source",
        key="source"
    )

    follow_up_date = st.date_input(
        "Follow-up Date",
        value=None,
        key="follow_up_date"
    )

    notes = st.text_area(
        "Notes",
        key="notes"
    )

    submitted = st.form_submit_button(
        "Add Application"
    )


if submitted:

    if not company or not role:
        st.error("Company and Role are required.")

    else:

        try:
            conn = get_connection()
            cursor = conn.cursor()

            query = """
                INSERT INTO applications
                (
                    company,
                    role,
                    job_url,
                    date_found,
                    date_applied,
                    status,
                    location,
                    salary,
                    source,
                    follow_up_date,
                    notes
                )
                VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            cursor.execute(
                query,
                (
                    company,
                    role,
                    job_url,
                    date_found,
                    date_applied,
                    status,
                    location,
                    salary,
                    source,
                    follow_up_date,
                    notes
                )
            )

            conn.commit()

            cursor.close()
            conn.close()

            st.success("Application added successfully!")

            # Tell Streamlit to clear the form
            st.session_state.reset_application_form = True

            # Rerun so the cleared values appear
            st.rerun()

        except Exception as e:
            st.error(f"Failed to add application: {e}")
# ============================================================
# APPLICATIONS
# ============================================================

st.header("My Applications")

try:

    conn = get_connection()

    df = pd.read_sql(
        """
        SELECT *
        FROM applications
        ORDER BY date_applied DESC;
        """,
        conn
    )

    conn.close()

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "job_url": st.column_config.LinkColumn(
                "Job URL",
                display_text="Open Job"
            )
        }
    )

except Exception as e:

    st.error(f"Failed to load applications: {e}")


# ============================================================
# UPDATE APPLICATION
# ============================================================

st.header("Update Application")

try:

    conn = get_connection()

    applications = pd.read_sql(
        """
        SELECT id, company, role
        FROM applications
        ORDER BY date_applied DESC;
        """,
        conn
    )

    conn.close()

    if len(applications) > 0:

        application_options = {
            f"{row['company']} - {row['role']} (ID: {row['id']})":
            row["id"]
            for _, row in applications.iterrows()
        }

        selected_application = st.selectbox(
            "Select application",
            list(application_options.keys())
        )

        selected_id = application_options[selected_application]

        # Get selected row
        conn = get_connection()

        selected_row = pd.read_sql(
            """
            SELECT *
            FROM applications
            WHERE id = %s;
            """,
            conn,
            params=(selected_id,)
        )

        conn.close()

        row = selected_row.iloc[0]

        with st.form("update_form"):

            new_company = st.text_input(
                "Company",
                value=row["company"]
            )

            new_role = st.text_input(
                "Role",
                value=row["role"]
            )

            new_url = st.text_input(
                "Job URL",
                value=row["job_url"] if pd.notna(row["job_url"]) else ""
            )

            current_status = row["status"]

            new_status = st.selectbox(
                "Status",
                STATUS_OPTIONS,
                index=STATUS_OPTIONS.index(current_status)
                if current_status in STATUS_OPTIONS
                else 0
            )

            new_location = st.text_input(
                "Location",
                value=row["location"] if pd.notna(row["location"]) else ""
            )

            new_salary = st.text_input(
                "Salary",
                value=row["salary"] if pd.notna(row["salary"]) else ""
            )

            new_source = st.text_input(
                "Source",
                value=row["source"] if pd.notna(row["source"]) else ""
            )

            new_notes = st.text_area(
                "Notes",
                value=row["notes"] if pd.notna(row["notes"]) else ""
            )

            update_button = st.form_submit_button(
                "Update Application"
            )

        if update_button:

            try:

                conn = get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE applications
                    SET
                        company = %s,
                        role = %s,
                        job_url = %s,
                        status = %s,
                        location = %s,
                        salary = %s,
                        source = %s,
                        notes = %s
                    WHERE id = %s;
                    """,
                    (
                        new_company,
                        new_role,
                        new_url,
                        new_status,
                        new_location,
                        new_salary,
                        new_source,
                        new_notes,
                        selected_id
                    )
                )

                conn.commit()

                cursor.close()
                conn.close()

                st.success("Application updated successfully!")

                st.rerun()

            except Exception as e:

                st.error(
                    f"Failed to update application: {e}"
                )

except Exception as e:

    st.error(
        f"Failed to load applications for editing: {e}"
    )


# ============================================================
# DELETE APPLICATION
# ============================================================

st.header("Delete Application")

try:

    conn = get_connection()

    applications = pd.read_sql(
        """
        SELECT id, company, role
        FROM applications
        ORDER BY date_applied DESC;
        """,
        conn
    )

    conn.close()

    if len(applications) > 0:

        delete_options = {
            f"{row['company']} - {row['role']} (ID: {row['id']})":
            row["id"]
            for _, row in applications.iterrows()
        }

        delete_selection = st.selectbox(
            "Select application to delete",
            list(delete_options.keys()),
            key="delete_select"
        )

        delete_id = delete_options[delete_selection]

        if st.button(
            "Delete Application",
            type="primary"
        ):

            try:

                conn = get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    """
                    DELETE FROM applications
                    WHERE id = %s;
                    """,
                    (delete_id,)
                )

                conn.commit()

                cursor.close()
                conn.close()

                st.success("Application deleted.")

                st.rerun()

            except Exception as e:

                st.error(
                    f"Failed to delete application: {e}"
                )

except Exception as e:

    st.error(
        f"Failed to load applications for deletion: {e}"
    )