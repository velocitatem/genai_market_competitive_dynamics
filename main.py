import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import timedelta
from collections import Counter
import os

def main():
    # ------------------------------
    # Data Loading and Preprocessing
    # ------------------------------
    try:
        df = pd.read_csv('data1.csv', encoding='utf8')
    except Exception as e:
        print("Error reading data1.csv:", e)
        return

    # Convert Date column to datetime objects and sort the data
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    print("Full chronological sequence of releases:")
    for _, row in df.iterrows():
        print(f"{row['Date'].strftime('%Y-%m-%d')}: {row['Company']} - {row['Model Name']}")

    # Create output directory for visualizations
    os.makedirs('outputs', exist_ok=True)

    # ------------------------------
    # Competitive Dynamics Analysis
    # ------------------------------
    # Analyze reaction times between competitors (responses within 90 days)
    reaction_times = []
    records = df.to_dict('records')
    n = len(records)
    for i in range(n):
        release = records[i]
        responses = []
        for next_release in records[i+1:]:
            if next_release['Company'] != release['Company']:
                diff_days = (next_release['Date'] - release['Date']).days
                if 0 <= diff_days <= 90:
                    responses.append({
                        'company': next_release['Company'],
                        'modelName': next_release['Model Name'],
                        'date': next_release['Date'],
                        'daysDiff': diff_days
                    })
        if responses:
            reaction_times.append({
                'trigger': release,
                'responses': responses
            })

    print("\nPotential competitive responses (within 90 days):")
    for rt in reaction_times:
        trigger = rt['trigger']
        print(f"After {trigger['Company']}'s {trigger['Model Name']} ({trigger['Date'].strftime('%Y-%m-%d')}):")
        for resp in rt['responses']:
            print(f"  - {resp['company']} released {resp['modelName']} after {round(resp['daysDiff'])} days")

    # Compute company reactivity metrics
    company_reactivity = {}
    for rt in reaction_times:
        for resp in rt['responses']:
            comp = resp['company']
            if comp not in company_reactivity:
                company_reactivity[comp] = {'totalResponses': 0, 'totalReactionTime': 0, 'reactions': []}
            company_reactivity[comp]['totalResponses'] += 1
            company_reactivity[comp]['totalReactionTime'] += resp['daysDiff']
            company_reactivity[comp]['reactions'].append({
                'inResponseTo': rt['trigger']['Company'],
                'reactionTime': resp['daysDiff']
            })
    for comp, data in company_reactivity.items():
        data['averageReactionTime'] = data['totalReactionTime'] / data['totalResponses']

    print("\nCompany reactivity metrics:")
    sorted_reactivity = sorted(company_reactivity.items(), key=lambda x: x[1]['averageReactionTime'])
    for comp, data in sorted_reactivity:
        print(f"{comp}: {data['totalResponses']} responses, avg reaction time: {round(data['averageReactionTime'])} days")
        response_targets = {}
        for r in data['reactions']:
            response_targets[r['inResponseTo']] = response_targets.get(r['inResponseTo'], 0) + 1
        if response_targets:
            most_target = max(response_targets.items(), key=lambda x: x[1])
            print(f"  Most frequently responds to: {most_target[0]} ({most_target[1]} times)")

    # Compute company influence metrics (who triggers responses)
    company_influence = {}
    for rt in reaction_times:
        trigger_comp = rt['trigger']['Company']
        if trigger_comp not in company_influence:
            company_influence[trigger_comp] = {'totalTriggeredResponses': 0, 'responders': {}}
        company_influence[trigger_comp]['totalTriggeredResponses'] += len(rt['responses'])
        for resp in rt['responses']:
            responder = resp['company']
            company_influence[trigger_comp]['responders'][responder] = company_influence[trigger_comp]['responders'].get(responder, 0) + 1

    print("\nCompany influence metrics (who triggers responses):")
    sorted_influence = sorted(company_influence.items(), key=lambda x: x[1]['totalTriggeredResponses'], reverse=True)
    for comp, data in sorted_influence:
        print(f"{comp} triggered {data['totalTriggeredResponses']} responses")
        top_responders = sorted(data['responders'].items(), key=lambda x: x[1], reverse=True)[:2]
        if top_responders:
            top_str = ', '.join([f"{res[0]} ({res[1]} times)" for res in top_responders])
            print(f"  Top responders: {top_str}")

    # Identify strong leader–follower relationships (follower responds at least twice to a leader)
    leader_follower_pairs = []
    for follower, data in company_reactivity.items():
        responses_by_trigger = {}
        for r in data['reactions']:
            responses_by_trigger[r['inResponseTo']] = responses_by_trigger.get(r['inResponseTo'], 0) + 1
        for leader, count in responses_by_trigger.items():
            if count >= 2:
                leader_follower_pairs.append({
                    'leader': leader,
                    'follower': follower,
                    'strength': count / data['totalResponses']
                })

    print("\nStrong leader-follower relationships:")
    for pair in sorted(leader_follower_pairs, key=lambda x: x['strength'], reverse=True):
        print(f"{pair['follower']} follows {pair['leader']} ({round(pair['strength']*100)}% of their responses)")

    # Quarterly release patterns
    df['YearQuarter'] = df['Date'].apply(lambda d: f"{d.year}-Q{((d.month - 1) // 3) + 1}")
    quarterly_counts = df['YearQuarter'].value_counts().sort_index()

    print("\nQuarterly release patterns:")
    for yq, count in quarterly_counts.items():
        print(f"{yq}: {count} releases")
        if count >= 3:
            quarter_data = df[df['YearQuarter'] == yq]
            companies_wave = quarter_data['Company'].tolist()
            print(f"  Wave of releases: {', '.join(companies_wave)}")
            dates = quarter_data['Date']
            range_days = (dates.max() - dates.min()).days
            if range_days <= 30:
                print(f"  Tight clustering: {count} releases within {range_days} days")

    # Identify potential preemptive releases (within 30 days before major player releases)
    major_players = ['OpenAI', 'Google', 'Microsoft', 'Anthropic', 'Meta']
    preemptive_releases = []
    for major in major_players:
        major_releases = df[df['Company'] == major]
        for idx, major_release in major_releases.iterrows():
            prior_releases = df[(df['Company'] != major) & (df['Date'] < major_release['Date']) &
                                (((major_release['Date'] - df['Date']).dt.days) <= 30)]
            if not prior_releases.empty:
                preemptive_releases.append({
                    'majorRelease': major_release,
                    'priorReleases': prior_releases.to_dict('records')
                })

    print("\nPotential preemptive releases (within 30 days before major releases):")
    for pr in preemptive_releases:
        major_rel = pr['majorRelease']
        print(f"Before {major_rel['Company']}'s {major_rel['Model Name']} ({major_rel['Date'].strftime('%Y-%m-%d')}):")
        for r in pr['priorReleases']:
            days_before = (major_rel['Date'] - pd.to_datetime(r['Date'])).days
            print(f"  - {r['Company']} released {r['Model Name']} {days_before} days before")

    # Identify potential counter-programming strategies (smaller companies releasing near major releases)
    smaller_players = [c for c in df['Company'].unique() if c not in major_players]
    counter_programming = {}
    for small in smaller_players:
        small_releases = df[df['Company'] == small]
        for idx, release in small_releases.iterrows():
            nearby_major = df[(df['Company'].isin(major_players)) &
                              (df['Date'].between(release['Date'] - pd.Timedelta(days=14),
                                                   release['Date'] + pd.Timedelta(days=14)))]
            if not nearby_major.empty:
                counter_programming.setdefault(small, []).append({
                    'ownRelease': release,
                    'nearbyMajorReleases': nearby_major.to_dict('records')
                })

    print("\nPotential counter-programming strategies:")
    for company, instances in counter_programming.items():
        if len(instances) >= 2:
            print(f"{company} has released {len(instances)} models near major player releases:")
            for instance in instances:
                own = instance['ownRelease']
                print(f"  {own['Model Name']} ({own['Date'].strftime('%Y-%m-%d')}) near:")
                for major in instance['nearbyMajorReleases']:
                    diff = (pd.to_datetime(major['Date']) - own['Date']).days
                    timing = f"{abs(diff)} days after" if diff > 0 else f"{abs(diff)} days before"
                    print(f"    - {major['Company']}'s {major['Model Name']} ({timing})")
    
    # Compute year-over-year growth in releases
    df['Year'] = df['Date'].dt.year
    yearly_counts = df['Year'].value_counts().sort_index()
    print("\nYear-over-year LLM release growth:")
    for year, count in yearly_counts.items():
        if year > yearly_counts.index[0]:
            prev_count = yearly_counts.get(year-1, 0)
            if prev_count > 0:
                growth = (count - prev_count) / prev_count * 100
                print(f"{year}: {count} releases ({growth:.1f}% growth from previous year)")
            else:
                print(f"{year}: {count} releases")
        else:
            print(f"{year}: {count} releases")
    
    # Compute average time between releases for each company
    company_release_intervals = {}
    for company in df['Company'].unique():
        company_releases = df[df['Company'] == company].sort_values('Date')
        if len(company_releases) >= 2:
            intervals = []
            for i in range(1, len(company_releases)):
                interval = (company_releases.iloc[i]['Date'] - company_releases.iloc[i-1]['Date']).days
                intervals.append(interval)
            company_release_intervals[company] = {
                'avg_interval': np.mean(intervals),
                'min_interval': np.min(intervals),
                'max_interval': np.max(intervals),
                'num_releases': len(company_releases)
            }
    
    print("\nCompany release cadence metrics:")
    sorted_intervals = sorted(company_release_intervals.items(), 
                             key=lambda x: x[1]['avg_interval'])
    for company, metrics in sorted_intervals:
        if metrics['num_releases'] >= 3:  # Only show companies with at least 3 releases
            print(f"{company}: releases every {metrics['avg_interval']:.0f} days on average" +
                 f" (range: {metrics['min_interval']}-{metrics['max_interval']} days, {metrics['num_releases']} releases)")

    # ------------------------------
    # Graphical Representations
    # ------------------------------
    sns.set(style="whitegrid")

    # 1. Timeline Scatter Plot of Releases
    plt.figure(figsize=(16, 10))
    companies = sorted(df['Company'].unique())
    company_to_num = {comp: i for i, comp in enumerate(companies)}
    df['CompanyNum'] = df['Company'].map(company_to_num)
    
    # Use a better color palette for many categories
    colors = sns.color_palette("hsv", len(companies))
    scatter = plt.scatter(df['Date'], df['CompanyNum'], s=120, c=[colors[i] for i in df['CompanyNum']], alpha=0.8)
    
    # Add model names as annotations
    for idx, row in df.iterrows():
        plt.annotate(row['Model Name'], 
                    (row['Date'], row['CompanyNum']),
                    xytext=(5, 0), 
                    textcoords='offset points',
                    fontsize=8,
                    rotation=45,
                    va='center')
    
    plt.yticks(list(company_to_num.values()), list(company_to_num.keys()))
    plt.xlabel("Release Date", fontsize=12)
    plt.ylabel("Company", fontsize=12)
    plt.title("Timeline of LLM Model Releases", fontsize=14, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/timeline_releases.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Bar Chart for Company Reactivity (Average Reaction Time)
    reactivity_df = pd.DataFrame([
        {'Company': comp,
         'Total Responses': data['totalResponses'],
         'Average Reaction Time': data['averageReactionTime']}
        for comp, data in company_reactivity.items()
    ])
    reactivity_df = reactivity_df[reactivity_df['Total Responses'] >= 3]  # Filter to companies with at least 3 responses
    reactivity_df = reactivity_df.sort_values('Average Reaction Time')
    plt.figure(figsize=(12,8))
    bars = sns.barplot(x='Average Reaction Time', y='Company', data=reactivity_df, palette="mako")
    
    # Add response count to bars
    for i, bar in enumerate(bars.patches):
        bars.text(
            bar.get_width() + 1, 
            bar.get_y() + bar.get_height()/2, 
            f"n={reactivity_df.iloc[i]['Total Responses']}", 
            va='center'
        )
    
    plt.title("Company Reactivity: Average Reaction Time (days)", fontsize=14, fontweight='bold')
    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Company", fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/company_reactivity.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Bar Chart for Company Influence (Total Triggered Responses)
    influence_df = pd.DataFrame([
        {'Company': comp, 'Triggered Responses': data['totalTriggeredResponses']}
        for comp, data in company_influence.items()
    ])
    influence_df = influence_df.sort_values('Triggered Responses', ascending=False)
    influence_df = influence_df.head(10)  # Top 10 most influential companies
    
    plt.figure(figsize=(12,8))
    sns.barplot(x='Triggered Responses', y='Company', data=influence_df, palette="rocket")
    plt.title("Company Influence: Total Triggered Responses", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Responses Triggered", fontsize=12)
    plt.ylabel("Company", fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/company_influence.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Quarterly Release Patterns
    quarter_df = quarterly_counts.reset_index()
    quarter_df.columns = ['YearQuarter', 'Release Count']
    plt.figure(figsize=(14,6))
    bars = sns.barplot(x='YearQuarter', y='Release Count', data=quarter_df, palette="viridis")
    plt.xticks(rotation=45)
    plt.title("Quarterly LLM Release Patterns (2020-2025)", fontsize=14, fontweight='bold')
    plt.xlabel("Year-Quarter", fontsize=12)
    plt.ylabel("Number of Releases", fontsize=12)
    
    # Annotate quarters with 3+ releases
    for i, bar in enumerate(bars.patches):
        if bar.get_height() >= 3:
            bars.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.1, 
                f"{bar.get_height():.0f}", 
                ha='center'
            )
    
    plt.tight_layout()
    plt.savefig('outputs/quarterly_releases.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Leader-Follower Network Graph
    G = nx.DiGraph()
    
    # Calculate node sizes based on total responses triggered (influence)
    node_sizes = {}
    for company, data in company_influence.items():
        node_sizes[company] = data['totalTriggeredResponses'] * 50 + 300  # Scale for visibility
    
    # Create node list with all companies (even if they don't have leader-follower relationships)
    for company in df['Company'].unique():
        if company not in node_sizes:
            node_sizes[company] = 300  # Default size for companies without measured influence
        G.add_node(company)
    
    # Add edges for leader-follower relationships
    for pair in leader_follower_pairs:
        leader = pair['leader']
        follower = pair['follower']
        weight = pair['strength']
        if weight >= 0.15:  # Only show stronger relationships (15%+ of responses)
            G.add_edge(leader, follower, weight=weight)
    
    plt.figure(figsize=(14,12))
    pos = nx.spring_layout(G, seed=42, k=0.3)  # Adjust k for better spacing
    
    # Draw nodes with varying sizes based on influence
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=[node_sizes.get(node, 300) for node in G.nodes()], 
        node_color='skyblue',
        alpha=0.7
    )
    
    # Draw edges with varying widths based on relationship strength
    edges = G.edges(data=True)
    weights = [edata['weight']*7 for (_, _, edata) in edges]  # Scale weights for visibility
    nx.draw_networkx_edges(
        G, pos, 
        arrowstyle='->', 
        arrowsize=20, 
        width=weights,
        edge_color='gray',
        alpha=0.6
    )
    
    # Draw labels with varying sizes
    label_font_sizes = {node: min(12 + node_sizes[node]/100, 16) for node in G.nodes()}
    for node, (x, y) in pos.items():
        plt.text(
            x, y, 
            node,
            fontsize=label_font_sizes.get(node, 10),
            ha='center', 
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
        )
    
    plt.title("Leader-Follower Relationships in the LLM Ecosystem", fontsize=16, fontweight='bold')
    plt.figtext(0.5, 0.01, "Node size represents influence (responses triggered); Edge width represents relationship strength", 
               ha='center', fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('outputs/leader_follower_network.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Heatmap of company responses
    response_matrix = {}
    for follower, data in company_reactivity.items():
        if follower not in response_matrix:
            response_matrix[follower] = {}
        for reaction in data['reactions']:
            leader = reaction['inResponseTo']
            response_matrix[follower][leader] = response_matrix[follower].get(leader, 0) + 1
    
    # Convert to DataFrame for visualization
    companies = sorted(set([comp for comp in df['Company'].unique()]))
    heatmap_data = []
    for follower in companies:
        row = {'Follower': follower}
        for leader in companies:
            row[leader] = response_matrix.get(follower, {}).get(leader, 0)
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df.set_index('Follower', inplace=True)
    
    # Filter to companies with at least one response
    active_companies = [company for company in companies if 
                       company in response_matrix or 
                       any(company in resp for resp in response_matrix.values())]
    
    heatmap_df = heatmap_df.loc[active_companies, active_companies]
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        heatmap_df, 
        annot=True, 
        cmap="YlGnBu", 
        fmt="d",
        linewidths=0.5,
        cbar_kws={"label": "Number of Responses"}
    )
    plt.title("LLM Company Response Patterns (Leader → Follower)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/response_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save leaders image for README
    plt.figure(figsize=(12, 8))
    influence_df = influence_df.head(8)  # Top 8 for better visibility
    sns.barplot(x='Triggered Responses', y='Company', data=influence_df, palette="rocket_r")
    plt.title("Top Industry Leaders: Companies That Trigger the Most Responses", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Responses Triggered", fontsize=12)
    plt.ylabel("Company", fontsize=12)
    plt.tight_layout()
    plt.savefig('leaders.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nAnalysis complete. Visualizations saved to 'outputs' directory.")

if __name__ == '__main__':
    main()