# SegFormer MVP API



# Project Structure

```txt
app/
│
├── main.py
├── requirements.txt
│
├── routes/
│   ├── crack_detection_routes.py
│   └── humidity_detection_routes.py
│
├── services/
│   ├── crack_detection_service.py
│   └── humidity_detection_service.py
│
├── models/crack/
    └──best_model_b3.pth
├── utils/
└── ...
```


# Install Requirements

```bash
pip install -r requirements.txt
```

---
# Convert Model
From the `app` folder:


```bash
python convert_model.py
```
# Run API

```bash
uvicorn main:app --reload
```

API will start on:

```txt
http://127.0.0.1:8000
```

---

# Swagger Documentation

Open:

```txt
http://127.0.0.1:8000/docs
```

---

# Crack Detection Endpoint

## Route

```txt
POST /crack-detection/predict
```

## Input Format

The API receives:

```txt
multipart/form-data
```

Example:

```txt
0 -> image1.jpg
1 -> image2.jpg
2 -> image3.jpg
```

Where:

* key = image id
* value = image file

---
# Response Format

The API returns:

```txt
multipart/mixed
```

## If crack detected

The API returns:

* image id
* segmented image

## If no crack detected

The API returns:

* image id only

---

# Humidity Detection Endpoint

## Route

```txt
POST /humidity-detection/predict
```

The behavior is the same as crack detection.

---

# Spring Boot Multipart/Mixed Parsing Example

The FastAPI API returns:

```txt
multipart/mixed
```

Spring Boot can parse the response and transform it into:

```java
Map<String, Object>
```

Where:

* key = image id
* value = segmented image bytes OR "no_crack"

---

## Maven Dependencies

```xml
<dependency>
    <groupId>org.apache.httpcomponents.client5</groupId>
    <artifactId>httpclient5</artifactId>
    <version>5.2.1</version>
</dependency>
```

---

## Spring Boot Example

```java
import org.apache.hc.client5.http.fluent.Request;

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

public class CrackDetectionClient {

    public static void main(String[] args) throws Exception {

        byte[] responseBytes = Request.post(
                "http://127.0.0.1:8000/crack-detection/predict"
        )
        .execute()
        .returnContent()
        .asBytes();

        String responseString =
                new String(
                        responseBytes,
                        StandardCharsets.ISO_8859_1
                );

        String boundary = extractBoundary(
                responseString
        );

        Map<String, Object> results =
                parseMultipart(
                        responseBytes,
                        boundary
                );

        for (String key : results.keySet()) {

            Object value = results.get(key);

            if (value instanceof byte[]) {

                System.out.println(
                        key + " -> crack detected"
                );

            } else {

                System.out.println(
                        key + " -> no crack"
                );
            }
        }
    }

    private static String extractBoundary(
            String response
    ) {

        int index = response.indexOf("--");

        int end = response.indexOf("
");

        return response.substring(
                index + 2,
                end
        );
    }

    private static Map<String, Object>
    parseMultipart(
            byte[] data,
            String boundary
    ) throws Exception {

        Map<String, Object> results =
                new HashMap<>();

        String delimiter =
                "--" + boundary;

        String content =
                new String(
                        data,
                        StandardCharsets.ISO_8859_1
                );

        String[] parts =
                content.split(delimiter);

        for (String part : parts) {

            if (
                    part.trim().isEmpty()
                    || part.contains("--")
            ) {
                continue;
            }

            String imageId = extractName(part);

            if (
                    part.contains(
                            "Content-Type: image/jpeg"
                    )
            ) {

                int bodyStart =
                        part.indexOf(
                                "

"
                        );

                String body =
                        part.substring(
                                bodyStart + 4
                        );

                byte[] imageBytes =
                        body.getBytes(
                                StandardCharsets.ISO_8859_1
                        );

                results.put(
                        imageId,
                        imageBytes
                );

            } else {

                results.put(
                        imageId,
                        "no_crack"
                );
            }
        }

        return results;
    }

    private static String extractName(
            String part
    ) {

        int start =
                part.indexOf("name=\"") + 6;

        int end =
                part.indexOf(
                        "\"",
                        start
                );

        return part.substring(
                start,
                end
        );
    }
}

