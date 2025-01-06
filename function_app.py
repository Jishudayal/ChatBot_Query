
import azure.functions as func
import logging
import json
from query_faiss import query_and_generate_response

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="HWaiAssistant")
def HWaiAssistant(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Processing HWaiAssistant HTTP request.")

    try:
        # Parse the request body
        req_body = req.get_json()
        user_query = req_body.get("query")
        session_id = req_body.get("session_id")
        user_id = req_body.get("user_id")
        client_id = req_body.get("client_id")
        dashboard_id = req_body.get("dashboard_id")

        # Validate required parameters
        if not user_query:
            return func.HttpResponse(
                json.dumps({"status": "error", "message": "Query parameter is missing."}),
                status_code=400,
                mimetype="application/json"
            )

        # Process the query
        response = query_and_generate_response(
            query=user_query,
            session_id=session_id,
            user_id=user_id,
            client_id=client_id,
            dashboard_id=dashboard_id,
        )

        # Handle clarifications
        if response.get("status") == "needs_clarification":
            return func.HttpResponse(
                json.dumps({
                    "status": "needs_clarification",
                    "clarifying_question": response["clarifying_question"],
                    "context_snippets": response["context_snippets"]
                }),
                mimetype="application/json",
                status_code=200
            )

        # Handle success responses
        if response.get("status") == "success":
            return func.HttpResponse(
                json.dumps(response), mimetype="application/json", status_code=200
            )

        # Handle errors
        return func.HttpResponse(
            json.dumps({"status": "error", "message": "An unexpected error occurred."}),
            status_code=500,
            mimetype="application/json"
        )

    except ValueError:
        return func.HttpResponse(
            json.dumps({"status": "error", "message": "Invalid JSON payload."}),
            status_code=400,
            mimetype="application/json"
        )
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return func.HttpResponse(
            json.dumps({"status": "error", "message": f"An error occurred: {str(e)}"}),
            status_code=500,
            mimetype="application/json"
        )
