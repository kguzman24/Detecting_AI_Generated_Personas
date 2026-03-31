# DS 4320 Project 1: Detecting AI-Generated Personas

**Summary:** The contents of this repository consists of a `data` folder that consists of the Instafake account data files `fakeAccountData.json` and `realAccountData.json`, as well as the `Human Faces` Dataset found on Kaggle. Additionally, the license and press release are displayed in the root repository, alongside the jupyter notebook that holds the code for the `Detecting AI Generated Personas` pipeline.

**Karen Guzman**

**Sae3gg**

**DOI:**

[**Press Release**](PressRelease.md)

[**Data**](https://1drv.ms/f/c/fbb5975494ef3dae/IgCypHVKJo39QaHkI1cbqYr8AaApb_AuER8r9EGxSNRTWD0?e=Y19Akp)

[**Pipeline**](Pipeline.ipynb)

[**LICENSE**](LICENSE)

## Problem Definition

### Problem

General Problem: Detecting AI-generated images/text/etc

Specific Problem: Detecting AI-generated personas in Social Media Profiles

### Rationale for refinement

Detecting AI-generated media is becoming harder, but the harm varies a lot by context. AI-generated personas are especially dangerous because victims face real financial or emotional damage. AI-written phishing emails can have more severe consequences than an AI-generated response to an interview application. This refinement on intentional  harm on people is important because it creates outlets for harm on real human life. Social media was chosen as the prime subject because of the growing cases of AI-generated personas.

### Motivation for this Project

This project is motivated by the need to detect AI-generated personas in social media. Fake profiles can have a variety of elements, including an AI-generated photo, which might not always be of a person. The image data paired with account metadata and profile activity can be very telling. A Dataset containing these elements would be essential to solving this problem.

### [**Press Release**](PressRelease.md)

Do You Really Know Your Followers?

## Domain Exposition

### Terminology
| # | Feature | Description |
|---|---------|-------------|
| 1 | **EXIF Data** | Metadata embedded in photo files by cameras/phones (timestamp, device model, GPS). AI images typically lack this entirely |
| 2 | **Laplacian Variance** | A measure of image sharpness; low values mean blurry |
| 3 | **Noise Level** | Random pixel-level variation in an image; real photos have natural camera noise, AI images often don't |
| 4 | **Saturation** | Intensity of colors in an image |
| 5 | **Pixel Distribution** | How color/brightness values are spread across an image |
| 6 | **Follower Ratio** | user_follower_count / user_following_count; fake accounts often follow many but have few followers back |
| 7 | **Profile Completeness** | How filled-out a profile is (has pic, has bio, post count); fake accounts tend to be sparse |

### Domain
My project lives at the intersection of cybersecurity and digital identity; it resides within the field of online harm detection, specifically social media. AI technology makes it easier to create convincing fake personas. This project directly addresses this potential danger, as AI-generated accounts could be used to deceive real users, potentionally enabling financial or emotional harm to others on a given platform. Social media safety would be at the core of this project.

### Background Reading
[Project 1 Background Reading Files](https://1drv.ms/f/c/fbb5975494ef3dae/IgDgNuMAxWXUSJDz-UbUBrbiAXMCOkcpQYHOGAGHodA1ifk?e=umWNUZ)

### Summary Table

| Title | Summary | Link |
|-------|---------|------|
| Generative AI personas considered harmful? Putting forth twenty challenges of algorithmic user representation in human-computer interaction | Examines the potential challenges that generative AI personas could have on stakeholder groups. There are multiple ways that harm can be done to society, for example, propagating biases, erosion of authentic user research practices, or misinformed design decisions based on these synthetically designed personas. | [Link](https://1drv.ms/t/c/fbb5975494ef3dae/IQB7wzEkzRlLSbT81sgV26jWAQbuTBvnnz3lQ3lqW7ukrVA?e=cDyJSv) |
| Digital Doppelgangers and AI Personas | Defines a digital doppelganger and how AI personas are created. This article covers mimicry mechanics, deepfake technology, and the effects of these personas on society. | [Link](https://1drv.ms/t/c/fbb5975494ef3dae/IQAi5MNvafWZQrCwQHI5HczRATFf3DloCNa-DmEp6iLvEAs?e=iQNx9R) |
| Risks of AI Mirror Social Media | Covers how AI is being used in social media; the article focuses on chatbots and their data collection. It also discusses the challenges of impulsivity and persuasion techniques used by chatbots that increase risk for both children and adults. | [Link](https://1drv.ms/t/c/fbb5975494ef3dae/IQDrdWoYYfgsTp97Ix5aOw4EAXvNFdvi31jo3PY9XEf41Ok?e=EcBGFJ) |
| Navigating the Risks of Artificial Intelligence on the Digital News Landscape | Focuses on the broader topic of disadvantages of AI in social media. Includes concerns of privacy, algorithm bias, job displacement, and more. | [Link](https://1drv.ms/t/c/fbb5975494ef3dae/IQDHxF6SZ5UDRoLoCmTuEMSaAfQ14gMkILduHszsvWPivO4?e=AuK5PP) |
| Social media platforms aren't doing enough to stop harmful AI bots, research finds | While this does not focus on AI-generated personas specifically, it discusses a broader topic of AI-generated content on social media: bots. Their study consisted of attempting to launch bots on various social media sites. | [Link](https://1drv.ms/t/c/fbb5975494ef3dae/IQDtrucQrTtNS5HRiDYOm8YdAVUNnxDevdEV9AMWQ_YU5OY?e=nFz5yx) |

## Data Creation

### Data Acquisition

The data acquisition process for this project was not terribly difficult. First, a Google search was done for datasets on AI generated pictures of human faces; the "Human Faces Dataset" was found and downloaded from Kaggle. The same was done for a dataset of curated account data; the instafake-dataset was found on a public GitHub repository and downloaded. Both were imported into the project Google Colab notebook.

### Code

| File | Description | Link |
|---|---|---|
| Design Project 1.ipynb | Loads face image dataset and account CSV data into Colab |[Detecting_AI_Generated_Personas](https://github.com/kguzman24/Detecting_AI_Generated_Personas)|

### Bias Identification

Bias could be introduced in this data collection process because both datasets are synthetic or curated. They are not drawn from real-world social media platforms. Real account profile pictures may not always feature faces, which is the focus of this project, meaning our model may not generalize well to real data.

### Bias Mitigation

These biases can be handled in our analysis clearly disclaiming that the data is synthetic and not representative of real social media accounts. Results should not be generalized beyond the scope of AI-generated face detection on similar synthetic datasets.

### Rationale
Because the timeline and scope of this poject is relatively short, I focused on only faces for profile pictures since it makes the most sense for an initial stage of detecting AI generated personas. The next stage in future work could account for other types of profile pictures.

\* note: fix "incomplete rationale"

## Metadata
### Schema


| | df_accounts |
|---|---|
| **PK** | *index (auto)* |
| | userFollowerCount   int NOT NULL |
| | userFollowingCount   int NOT NULL |
| | userBiographyLength   int NOT NULL |
| | userMediaCount   int NOT NULL |
| | userHasProfilPic   int NOT NULL |
| | userIsPrivate   int NOT NULL |
| | usernameDigitCount   int NOT NULL |
| | usernameLength   int NOT NULL |
| | isFake   int NOT NULL |
| | label   str NOT NULL |
| | profile_pic   str NOT NULL |

### Data Tables \*change to links to csv files

| Table | Description | File |
|---|---|---|
| account_profile.csv | Contains details about the account profile, includes `account_id`,`bio_length`, `has_profile_pic`,                      `is_private`, `username_digit_count`, and `username_length`. | [account_profile.csv](https://1drv.ms/x/c/fbb5975494ef3dae/IQAT93xX0CLnQa2unLUdyQlsAVZp3cJMPhRHUWHxt2fvmbU?e=ckr5Td) |
| account_stats.csv | Contains details on an accounts metrics including `account_id`, `follower_count`,                            `following_count`, and `media_count`. | [accounts_stats.csv](https://1drv.ms/x/c/fbb5975494ef3dae/IQBBVnpSZhLRRq6ZKVXX8otDASSgOk5XOm4P0PYmzzXPfU8?e=r2RK64) |
|accounts.csv | Contains details on whether the account is fake as well as the profile picture id: `account_id`,`label`,`isFake`,and `profile_pic_id` [accounts.csv](https://1drv.ms/x/c/fbb5975494ef3dae/IQBSQKV6lSDDTaim1uh_yZ6EAXT5mVUuj8jVqtyG5S_jV9w?e=kWyBuT) |
|face_images.csv | Contains image details from the Human Faces Dataset on Kaggle: `image_id`,`filename`, and`face_type` | (face_images.csv)[https://1drv.ms/x/c/fbb5975494ef3dae/IQCZmR__SYAqTIsnhEN5buN6AdLG5yWIHaetnjRmiQbL4g0?e=K7l36c]

### Data Dictionary

| Name | Data Type | Description | Example | Uncertainty|
|---|---|---|---|---|
| userFollowerCount | int | Number of followers | 304 | 0 uncertainty if we are able to see the user follower count from a profile in real time. The uncertainty of this metric may increase as the user gains or loses followers.|
| userFollowingCount | int | Number of accounts following | 449 |  0 uncertainty if we are able to see the user following count from a profile in real time. The uncertainty of this metric may increase as time passes and the user unfollows or follows more people.|
| userBiographyLength | int | Character length of bio | 22 | 0 uncertainty if we are able to see the bio from a profile in real time. The uncertainty of this metric may increase as time passes and changes could be made to a profile's bio.|
| userMediaCount | int | Number of posts | 20 |  0 uncertainty if we are able to see user's public media count. The uncertainty of this metric may increase as time passes and posts are made or deleted.|
| userHasProfilPic | int (binary) | Whether account has profile pic (1=yes) | 1 | 0 uncertainty if we are able to see the user profile in real time. The uncertainty of this metric may increases with time, as the user could remove or add a picture to their profile.|
| userIsPrivate | int (binary) | Whether account is private (1=yes) | 0 | 0 uncertainty if we are able to see if the user has a public account or if a follow request has to be made. The uncertainty of this metric may increase as time passes and account privacy settings change.|
| usernameDigitCount | int | Number of digits in username | 0 | 0 uncertainty if we are able to see the digits in a username through the profile. The uncertainty of this metric may increase as time passes and changes are made to the account.|
| usernameLength | int | Total length of username | 11 | 0 uncertainty if we are able to see see the username of an account. The uncertainty of this metric may increase as time passes and changes are made to the account.|
| isFake | int (binary) | Original fake label from InstaFake (1=fake) | 1 | A moderate amount of uncertainty when our model declares that an account is fake.|
| label | string | Human-readable label | fake |
| profile_pic | string | Filename of assigned face image | ai_001.jpg | A moderate amount of uncertainty when our model declares that an account is real.
