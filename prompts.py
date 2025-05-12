#!/usr/bin/env python3
"""
Prompt templates for the food insecurity synthetic dataset generator.

This file contains the prompts used for:
1. Generating knowledge triplets
2. Creating articles from structured truth data
3. Generating non-food-related articles
"""

# Prompt to generate knowledge triplets for a specific category and country
TRIPLETS_GENERATION_PROMPT = """
Generate between 1 and 5 realistic cause-effect relationships (knowledge triplets) about food insecurity 
in {country}, specifically related to {category}. Make sure to randomly select the number of triplets.

Each triplet should have:
- subject: A specific cause or factor
- relation: The type of relationship
- effect: The impact on food security

Format as valid JSON like this:
{{
  "knowledge_triplets": [
    {{
      "subject": "cause or factor",
      "relation": "type of relationship",
      "effect": "effect on food security"
    }},
    ...
  ]
}}

Make the triplets realistic, detailed, and specific to {country} and {category} and geopolitical, social, and economic context of the country in the last 10 years. Reference true events and data.
Output ONLY the JSON.
"""

# Prompt to generate non-food-related articles
NON_FOOD_ARTICLE_PROMPT = """
Write a realistic news article in {language} about {topic} (not related to food insecurity) in {district}, {region}, {country}.

Format the article as a professional news piece with:
- A headline
- Dateline ({news_source}, {date})
- 300-{max_words} words of content
- Include quotes from relevant people
- Make the article realistic and factual in tone
- TOPICS: politics, economics, culture, science, technology, sports, entertainment, etc.

The article should be completely unrelated to food insecurity, hunger, famine, or agricultural issues.

Write ONLY the article text as it would appear in a news publication. Start the article with <start_of_article> and end with <end_of_article>.
"""


# Improved prompt to generate an article from structured truth data
ARTICLE_FROM_TRUTH_PROMPT = """
Write a realistic news article in {language} about food insecurity in {district}, {region}, {country}.

The article should present {category} as a primary cause of food insecurity, but incorporate all these cause-effect relationships naturally throughout the narrative:
{triplets_text}

IMPORTANT WRITING GUIDELINES:
- Write like a professional journalist covering a humanitarian crisis
- DO NOT structure the article as a formula where each paragraph addresses exactly one triplet
- Instead, weave causes and effects organically throughout the article as real journalists do
- Some paragraphs should blend multiple related causes or effects
- A significant cause might be developed across several paragraphs from different angles
- Use human stories and personal experiences can be used to illustrate abstract concepts where appropriate
- Include relevant statistics and expert assessments where appropriate
- Create natural transitions between topics rather than abrupt shifts between causes
- Maintain factual accuracy, situate events in real social, political, and economic context, and reference real events, data, and statistics

FORMAT REQUIREMENTS:
- A compelling headline that captures the core issue
- Dateline ({news_source}, {date})
- 300-{max_words} words of content

Write ONLY the article text as it would appear in a news publication. Start the article with <start_of_article> and end with <end_of_article>.

As an example, the following ground-truth information:


{
  "article_id": "bd73e28c-f190-4a5d-9e12-937c8d463e1a",
  "language": "EN",
  "word_count": 1242,
  "metadata": {
    "publication_date": "2024-06-28T00:00:00",
    "country": "Palestine",
    "region": "Middle East",
    "district": "Gaza Strip",
    "source": "Financial Times"
  },
  "food_insecurity": {
    "is_relevant": true,
    "category": "conflict and violence"
  },
  "knowledge_graph": {
    "triplets": [
      {
        "subject": "Israeli military offensive and restrictions",
        "relation": "prevents",
        "effect": "food and humanitarian goods reaching hungry population"
      },
      {
        "subject": "Border closure (Rafah crossing with Egypt)",
        "relation": "blocks",
        "effect": "commercial food imports and aid entry into Gaza"
      },
      {
        "subject": "War economy price inflation",
        "relation": "makes",
        "effect": "basic foods unaffordable for most families"
      },
      {
        "subject": "Destruction of banking infrastructure",
        "relation": "prevents",
        "effect": "families from accessing money to purchase available food"
      },
      {
        "subject": "Limited flour availability from aid agencies",
        "relation": "serves as",
        "effect": "primary barrier against mass starvation"
      }
    ]
  }
}

Would generate the following article: 

<start_of_article>"Ramy al-Mutawaq was immensely proud of his well-stocked grocery shop in Jabalia in northern Gaza. Before the war, whenever he felt unhappy, he could lift his spirits by simply sweeping his eyes around its shelves stacked with merchandise.
"I always had at least between $5,000 and $6,000 worth of stock, and every day I would bring in new supplies," Mutawaq said. "I had everything one could desire, including all kinds of chocolate and instant coffee."
Now, like many other grocers and supermarkets in the bombed-out wastelands of northern Gaza, Mutawaq's shop is seldom open. There is nearly nothing to sell -- and after war laid waste to the strip and its economy, even tinned peas or beans are too expensive for most people. Starvation is setting in, and Mutawaq's ample stocks are just a memory.
As Israel pushes on with its offensive in the strip, the hostilities -- along with restrictions imposed by Israel, the closure of the Rafah crossing with Egypt and looting by gangs inside the devastated territory -- mean the flow of food and other humanitarian goods to the hungry, exhausted population has slowed to a trickle. International pressure on Israel to ensure greater aid provision has had limited effect.
Mutawaq has opened his shop only intermittently since the war began in October, sometimes just to sell soap and cleaning materials.
He gets his merchandise from bigger traders, he said, who often buy coupons from aid recipients entitled to humanitarian goods. Food looted from aid trucks also reaches the market, but recent weeks have been especially dire.
"There is no merchandise so for now I am closed up and sitting at home," he said. "People can't afford food. They make do with anything -- some instant coffee, or tea, or a bit of bread with thyme."
Price volatility in the enclave's closed wartime economy has meant that acquiring stock comes with significant risks.
"I cannot stock big quantities because prices keep on fluctuating," he said. "I can't get a whole 50kg sack of sugar, for instance, because the price per kilo could go down if aid agencies bring in sugar."
Sugar was selling for $20 a kilo, Mutawaq said, up from 53 cents before the war.
Mutawaq's three children survive on the same meagre diet that is barely keeping other Palestinians alive in Gaza's north. He and other parents feed their children vitamins from pharmacies that are still open, hoping to shore up their health. His eight-month-old has to make do with breadsticks or biscuits dipped in warm water.
Just over a fifth of Gaza's population, or about half a million people, face the most severe, "catastrophic" level of hunger, in which "starvation, death, destitution and extremely acute malnutrition levels are evident", according to an assessment this month by the Integrated Food Security Phase Classification, an international hunger-monitoring mechanism.
More than half of households have had to exchange clothes for money to buy food, while a third resorted to picking up trash to sell, the organisation said; more than a fifth went entire days and nights without eating.
A $230mn US-built floating pier, intended as a direct route to deliver aid brought by sea, has made little difference after repeatedly going out of service and facing problems distributing what supplies do reach shore.
For part of April and May, when Israel allowed trucks of commercial goods into the north, fresh produce and many basic commodities reappeared. Mutawaq and other shopkeepers reopened their stores. But this ended when Israel's offensive in north Gaza resumed in May.
"We opened for about a month during that period," said Essam Aboul Hosna, owner of a shop on the outskirts of Jabalia. "We had rice, lentils and canned goods. Merchandise for the private sector was getting in. But it all suddenly disappeared."
He said wholesalers had run out of goods, while food "smuggled" from the south was too expensive: the consumer price of a kilo of tomatoes had reached $26 and green peppers $64.
"Neither merchant nor consumer can afford this," he said.
Flour from aid agencies has been the only thing standing between many people and starvation. "Here in the north we have eaten animal fodder made of maize and even soya, but now we have flour," Aboul Hosna said.
Gazans trying to secure food have been stymied not only by prohibitive prices but also an inability -- even among those with bank accounts -- to access cash.
Court official Majeda al-Adham, a mother of eight, has been unable to cash her salary, which is transferred to her account from the Palestinian Authority in the West Bank, since March. Bank branches have been bombed or robbed, and most cash machines do not function. Al-Adham once had to pay a 25 per cent commission to an exchange bureau to get banknotes against a digital transfer from her account.
After repeatedly being forced to flee, al-Adham's family returned in recent weeks to Jabalia, to a house with windows and doors blown out by the explosions that levelled much of the once-congested district.
Since then, she has been growing some courgettes and aubergines to feed the family. Sometimes they just eat tomato sauce or bread with thyme.
"What can we do, given how food is so expensive? We all help each other and some of us don't eat, so there will be enough for the children," said Adham. "You go to the shops, but there is nothing there.
"A can of peas costs Shk20 ($5.30). I would need five cans to cook a meal for the children."
Taisir al-Tanna, a vascular surgeon at the al-Ahly hospital in Gaza City, lives in al-Tuffah, not far from Jabalia. He said the markets had no vegetables, fruits or meat, only tinned food.
A rare carton of 30 eggs costs $70 to $80, he said.
"Food is very limited, and when you eat you are never full. It is just something to keep you going," he said. "There is flour, however, at $12 for a 25kg sack."
There is no fuel or cooking gas, so his wife and children burn firewood to bake bread. "It is as if we have been taken back 150 years," he said.
The World Food Programme said on June 23 it had reopened one of Jabalia's two bakeries, producing bread for 3,000 families. Matthew Hollingworth, the organisation's country director for Palestine, said in a video filmed at the bakery that while it helped achieve "some small levels of food security", it was still "essential for commercial fresh food to enter northern Gaza".
He said every child he met in Gaza was dreaming "of eating vegetables . . . [and] meat [and] they're sick of eating aid [such as tinned food], even though it's keeping them alive. But it's barely a life."
International attempts to broker a ceasefire that would allow more supplies into the enclave have so far failed. Israel has said it is shifting to a lower-intensity phase of conflict, but after nine months of war, Al-Mutawaq has given up on returning to the modest prosperity he previously achieved.
"All our ambitions for the future have been dashed," he said. "There is nothing to be optimistic about."<end_of_article>


Use this example to undestand the complexity, nuance, and style of a realistic article. Note how paragraphs 3-4 blend multiple causes (military action, border closures, and economic impacts) rather than treating each cause separately
NOTE: your article should be more concise than the example while maintaining the same organic integration of causes and effects.
"""