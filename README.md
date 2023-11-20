# Scope+: Open Source Generalizable Architecture for Single-cell Atlases

<p align="center">
  <img width="500"  src="/screenshots/Logo_gradient.png">
</p>

Scope+ is a modern, scalable and generalized single-cell atlas portal archictecture. The architecture has been applied to 5 million single-cell COVID-19 immune and blood cells data with realization in Covidscope web portal (https://www.covidsc.d24h.hk/). 

## Tutorial
The tutorial for implementation of Scope+ to adapt user's own atlas datasets is available at: [https://hiyin.github.io/scopeplus-user-tutorial/](https://hiyin.github.io/scopeplus-user-tutorial/)

## Introduction
Single-cell technologies have been widely used to investigate the cellular response and mechanisms for COVID-19. Due to the increasing large-scale single-cell studies, many integrated datasets have been made available, and there is a lack of fast and easy access tools to examine such resources. These large-scale data poses a data science challenge of effective accessing, organizing, sharing and interpretation of integrated datasets. There are many individual platforms that try to host the single-cell atlas, each with their own limitations. We believe that there should be a generalizable architecture for development of single-cell atlas, enhancing reproducible research.  Here, we develop an open source, highly optimized and scalable architecture, named Scope+, to allow quick access, meta-analysis and cell-level selection of the atlas data. We applied this architecture to our well-curated Covid-19 resources, releasing it as a portal i.e. Covidscope (https://covidsc.d24h.hk/) that facilitates effective meta-analytical downstream analysis for 5 million single-cell COVID-19 transcriptomic data from 20 studies globally. Covidscope is not limited to our own Covid-19 atlas, it can be adapted and reimplemented to any other single-cell transcriptomic atlas or integrated datasets.  

## Innovation
Scope+ addresses the big data challenges associated with fast access, cell-level querying and meta-analyses of 12.7B matrix count and ~5M metadata in a web application with innovative architecture design. We address the challenge   with three key innovations: (i) server-side rendering of metadata via a fast pagination technique; (ii) novel database optimisation strategies to allow extremely fast retrieval of the large count matrix; and (iii) a novel architecture design using enabling users to filter, visualize and large queried data concurrently. Thus, Scope+
allows real-time data subsetting and visualizations of the large single cell expression data matrix based on user queries using patient or cell-level characteristics with real-time dynamic visualizations. 

## Implication
Existing single-cell atlas portal allow dataset-level browsing, no integrative cell-level browsing, while Scope+ provide efficient access, cell-level categorical selections and filtering of cells for meta-analysis for users who wish to use and adopt Scope+ to their own atlas data. Single cell gene expression data is sparse high-dimentional, the data query, retrieval and data visualization in Scope+ at cell-level granularity is accelerated by various computational optimization strategies. 

The development timeline for single cell atlas portal is around 1 to 2 years, with the help of Scope+, it can be streamlined to a weeke or two. Ultimately, Scope+ can translate any single cell atlases into a openly accessible data web portal for reproducible science, and making single-cell research data avaialble for community timely, this is particularly important for pandemics. 

## Liscence
The software architecture is open-sourced in this repository and for use under the MIT License.

