"""Hand-crafted dataset: Spring Boot incident triage.

Each example is a realistic production incident (log line / stack trace excerpt)
labeled with a root-cause category, a severity and a short remediation hint.

24 examples total: 14 train / 10 val — deliberately tiny so the optimization
runs in this repo cost cents, not dollars.
"""

import dspy

ROOT_CAUSES = [
    "config-misconfiguration",
    "dependency-conflict",
    "connection-pool-exhaustion",
    "serialization-error",
    "memory-pressure",
    "thread-starvation",
]

# (incident_report, root_cause, severity, action)
_RAW = [
    # --- config-misconfiguration -------------------------------------------------
    ("Boot fails: 'Could not resolve placeholder DB_URL in value ${db.url}' on startup",
     "config-misconfiguration", "P1", "Set the DB_URL environment variable or a default in application.yml"),
    ("Config server returns 404 for application/prod.yml; service boots with defaults and writes to the wrong DB",
     "config-misconfiguration", "P1", "Fix the config-server branch/path and restart the service"),
    ("Actuator /health returns DOWN: 'Management port 9090 already in use' after moving management.server.port",
     "config-misconfiguration", "P3", "Free port 9090 or update management.server.port in the prod profile"),
    ("Profiles collide: application-prod.yml sets hibernate.ddl-auto=update, prod DB grew an extra audit table overnight",
     "config-misconfiguration", "P2", "Set ddl-auto=validate in prod and manage schema with migrations"),
    # --- dependency-conflict -----------------------------------------------------
    ("NoSuchMethodError com.fasterxml.jackson.databind.ObjectMapper.coercionConfigFor at app startup",
     "dependency-conflict", "P1", "Align jackson-databind versions with the Spring Boot BOM and re-build"),
    ("java.lang.ClassNotFoundException: org.apache.kafka.clients.consumer.Consumer after upgrading a starter",
     "dependency-conflict", "P1", "Pin kafka-clients to the version required by spring-kafka"),
    ("Guava 18 and 32 on the classpath at once; StopWatch$StopwatchStyle cannot be cast after adding a vendor SDK",
     "dependency-conflict", "P2", "Force one Guava version via dependency management and run dependency:tree"),
    # --- connection-pool-exhaustion ----------------------------------------------
    ("HikariPool-1 - Connection is not available, request timed out after 30000ms under evening peak load",
     "connection-pool-exhaustion", "P1", "Raise maximum-pool-size and fix the transaction leaking a connection"),
    ("Pools starve: a @Transactional method calls a slow external REST API while holding the DB connection",
     "connection-pool-exhaustion", "P2", "Move the remote call outside the transaction boundary"),
    ("Connections leak: leakDetectionThreshold logs 'Connection leak detection: owner thread holds connection for 180s'",
     "connection-pool-exhaustion", "P2", "Audit code paths that open connections without closing them"),
    # --- serialization-error -----------------------------------------------------
    ("HttpMessageNotReadableException: JSON parse error on Instant field '2024-13-45T99:00:00Z' from the partner API",
     "serialization-error", "P3", "Validate the partner payload and add a lenient deserializer for dates"),
    ("InvalidDefinitionException: No serializer found for class Employee$Builder and no properties discovered",
     "serialization-error", "P3", "Annotate the DTO (not the builder) with @JsonProperty or drop Lombok @Builder from the API type"),
    ("Caused by: com.fasterxml.jackson.databind.exc.MismatchedInputException: Expected ARRAY but was OBJECT for field 'items'",
     "serialization-error", "P2", "Regenerate the client from the current OpenAPI spec"),
    # --- memory-pressure ---------------------------------------------------------
    ("java.lang.OutOfMemoryError: Java heap space after loading a 2GB CSV into a List<String> for validation",
     "memory-pressure", "P1", "Stream the file line by line instead of buffering it in memory"),
    ("GC overhead limit exceeded; full GCs take 40s each, heap dump shows a static Map cache at 6GB",
     "memory-pressure", "P1", "Bound the cache with Caffeine maximumSize and a TTL"),
    ("Pod OOMKilled by Kubernetes at 1Gi limit even though heap usage is only 400MB",
     "memory-pressure", "P2", "Cap -XX:MaxRAMPercentage=70 or raise the container memory limit"),
    # --- thread-starvation -------------------------------------------------------
    ("Tomcat threads exhausted: 200/200 busy, all blocked on a synchronized block in LegacyRateLimiter",
     "thread-starvation", "P1", "Replace the synchronized block with a striped lock or a non-blocking rate limiter"),
    ("@Scheduled job every 10s blocks all default scheduler threads; health checks never run",
     "thread-starvation", "P2", "Give long jobs a dedicated executor via a TaskScheduler pool"),
    ("CompletableFuture chain uses the common ForkJoinPool and starves parallel streams app-wide",
     "thread-starvation", "P2", "Run async work on a dedicated bounded executor"),
    ("Deadlock detected in thread dump: two transactions lock accounts in opposite order",
     "thread-starvation", "P1", "Acquire locks in a canonical order (sort account ids) in every code path"),
    # --- a few mixed extras --------------------------------------------------------
    ("Startup error: 'Parameter 0 of constructor in PaymentService required a bean of type FraudClient that could not be found'",
     "config-misconfiguration", "P2", "Add @ComponentScan for the client package or declare the Feign client explicitly"),
    ("Intermittent StackOverflowError deep in recursion while flattening deeply nested JSON config trees",
     "memory-pressure", "P3", "Rewrite the recursion iteratively with an explicit stack"),
    ("Retry storm: 40 threads wait on the same Hikari connection while each caller retries 5 times with no backoff",
     "connection-pool-exhaustion", "P1", "Add exponential backoff with jitter and a circuit breaker"),
    ("After adding lombok-mapstruct-binding, MapStruct mappers generate empty impls and NPEs surface at runtime",
     "dependency-conflict", "P2", "Pin lombok-mapstruct-binding to the documented version in annotationProcessorPaths"),
]


def _to_example(row, with_inputs=True):
    report, root_cause, severity, action = row
    ex = dspy.Example(
        incident_report=report,
        root_cause=root_cause,
        severity=severity,
        action=action,
    ).with_inputs("incident_report")
    return ex


def build_datasets():
    rows = list(_RAW)
    trainset = [_to_example(r) for r in rows[:14]]
    valset = [_to_example(r) for r in rows[14:]]
    return trainset, valset
