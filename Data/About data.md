Based on the `CoSupplyChainData.csv` file, here is a structured and detailed data description formatted for a GitHub `README.md`. It highlights the supply chain dimensions, key features, and the target variables used for your GNN model.

---

## Dataset Overview

The dataset used in this project is the **DataCo Smart Supply Chain for Big Data Analysis**, which covers structured supply chain data including registered users, locations, and item information. It is particularly valuable for modeling complex relationships between orders, products, and shipping risks.

### **Core Dimensions**

The data is categorized into several functional areas of a supply chain:

| Dimension | Key Features |
| --- | --- |
| **Shipping** | Shipping Mode, Days for Shipping (Real vs. Scheduled), Delivery Status. |
| **Customer** | Segment, City, State, Country, Zipcode. |
| **Product** | Category Name, Product Price, Product Card ID. |
| **Financial** | Benefit per Order, Sales per Customer, Order Profit, Item Total. |
| **Temporal** | Order Date, Shipping Date (used for lead time analysis). |

---

## Key Data Attributes

### **1. Target Variables for Risk Assessment**

* **`Late_delivery_risk`**: A binary flag (0 or 1) indicating if a delivery is likely to be delayed. This serves as the primary label for the Early Warning System.
* **`Delivery Status`**: Categorical status including *Late delivery*, *Advance shipping*, *Shipping on time*, and *Shipping canceled*.

### **2. Geospatial & Network Features**

* **`Order Region` / `Market**`: Geographical clusters (e.g., Southeast Asia, Western Europe, Central America) used to build the graph nodes.
* **`Latitude` / `Longitude**`: Precise coordinates used to calculate spatial distances between nodes in the GNN.

### **3. Transactional Metrics**

* **`Order Item Profit Ratio`**: Critical for identifying high-impact nodes where a disruption would cause the most financial damage.
* **`Sales`**: Total sales volume per order, used to weight the importance of specific edges in the supply chain graph.

---

## Data Preprocessing & Graph Construction

To make this data suitable for **Graph Neural Networks (GNNs)**, the following transformations are applied:

1. **Node Definition**: Entities (Suppliers/Regions) are represented as nodes.
2. **Edge Creation**: Edges are formed based on:
* **Flow**: Common product categories moving between regions.
* **Similarity**: Shared shipping modes or geographical proximity.


3. **Feature Scaling**: Financial columns (Sales, Profit) are normalized to ensure stable gradient descent during model training.
4. **Temporal Encoding**: Order dates are converted into cyclical features (Month, Day of week) to capture seasonal supply chain bottlenecks.

---

## Data Summary

* **Format**: Comma Separated Value (CSV)
* **Size**: ~180,000 rows
* **Coverage**: Global supply chain operations including 5 main markets (Pacific Asia, Europe, Africa, LATAM, North America).

> **Note:** For the GNN implementation, the tabular data is transformed into an adjacency matrix  and a feature matrix , where  contains the encoded tensor attributes of each supplier/region node.
